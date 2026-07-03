#include <mitsuba/core/plugin.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/render/volume.h>

NAMESPACE_BEGIN(mitsuba)

MI_VARIANT Medium<Float, Spectrum>::Medium()
    : JitObject<Medium>(""),
      m_is_homogeneous(false),
      m_has_spectral_extinction(true),
      m_majorant_resolution_factor(0),
      m_majorant_factor(1.01f),
      m_majorant_grid_res(0u) {
}

MI_VARIANT Medium<Float, Spectrum>::Medium(const Properties &props)
    : JitObject<Medium>(props.id()) {
    for (auto &prop : props.objects()) {
        if (PhaseFunction *phase = prop.try_get<PhaseFunction>()) {
            if (m_phase_function)
                Throw("Only a single phase function can be specified per medium");
            m_phase_function = phase;
        }
    }
    if (!m_phase_function) {
        // Create a default isotropic phase function
        m_phase_function =
            PluginManager::instance()->create_object<PhaseFunction>(Properties("isotropic"));
    }

    m_sample_emitters = props.get<bool>("sample_emitters", true);

    /* Majorant supergrid: 0 disables it (single global majorant). Values > 0
       coarsen the extinction volume's native resolution by this factor. */
    m_majorant_resolution_factor =
        (uint32_t) props.get<int>("majorant_resolution_factor", 0);
    m_majorant_factor = props.get<ScalarFloat>("majorant_factor", 1.01f);
    m_majorant_grid_res = ScalarVector3u(0u);
}

MI_VARIANT Medium<Float, Spectrum>::~Medium() { }

MI_VARIANT void Medium<Float, Spectrum>::traverse(TraversalCallback *cb) {
    cb->put("phase_function", m_phase_function, ParamFlags::Differentiable);
}

MI_VARIANT
typename Medium<Float, Spectrum>::MediumInteraction3f
Medium<Float, Spectrum>::sample_interaction(const Ray3f &ray, Float sample,
                                            UInt32 channel, Mask active) const {
    MI_MASKED_FUNCTION(ProfilerPhase::MediumSample, active);

    // initialize basic medium interaction fields
    MediumInteraction3f mei = dr::zeros<MediumInteraction3f>();
    mei.wi          = -ray.d;
    mei.sh_frame    = Frame3f(mei.wi);
    mei.time        = ray.time;
    mei.wavelengths = ray.wavelengths;

    auto [aabb_its, mint, maxt] = intersect_aabb(ray);
    aabb_its &= (dr::isfinite(mint) || dr::isfinite(maxt));
    active &= aabb_its;
    dr::masked(mint, !active) = 0.f;
    dr::masked(maxt, !active) = dr::Infinity<Float>;

    mint = dr::maximum(0.f, mint);
    maxt = dr::minimum(ray.maxt, maxt);

    UnpolarizedSpectrum combined_extinction;
    Float sampled_t;
    Mask valid_mi;
    if (has_majorant_grid()) {
        /* Spatially-varying majorant: sample against the piecewise-constant
           supergrid with a DDA traversal. The local majorant of the cell
           containing the sample is reported as `combined_extinction`; the
           tr/pdf *ratio* returned by transmittance_eval_pdf() remains exact
           under this convention (the accumulated-optical-depth exponentials
           cancel), which is the only way existing integrators consume it. */
        auto [dda_t, local_majorant, dda_valid] =
            sample_interaction_dda(ray, mint, maxt, sample, active);
        combined_extinction = UnpolarizedSpectrum(local_majorant);
        sampled_t           = dda_t;
        valid_mi            = dda_valid && (sampled_t <= maxt);
        DRJIT_MARK_USED(channel);
    } else {
        combined_extinction = get_majorant(mei, active);
        Float m             = combined_extinction[0];
        if constexpr (is_rgb_v<Spectrum>) { // Handle RGB rendering
            dr::masked(m, channel == 1u) = combined_extinction[1];
            dr::masked(m, channel == 2u) = combined_extinction[2];
        } else {
            DRJIT_MARK_USED(channel);
        }
        sampled_t = mint + (-dr::log(1 - sample) / m);
        valid_mi  = active && (sampled_t <= maxt);
    }
    mei.t           = dr::select(valid_mi, sampled_t, dr::Infinity<Float>);
    mei.p           = ray(sampled_t);
    mei.medium      = this;
    mei.mint        = mint;

    std::tie(mei.sigma_s, mei.sigma_n, mei.sigma_t) =
        get_scattering_coefficients(mei, valid_mi);
    mei.combined_extinction = combined_extinction;
    return mei;
}

MI_VARIANT void
Medium<Float, Spectrum>::update_majorant_grid(const Volume *volume,
                                              ScalarFloat scale) {
    if (m_majorant_resolution_factor == 0)
        return;

    ScalarVector3u res;
    DynamicBuffer<Float> cells =
        volume->local_majorants(m_majorant_resolution_factor, res);
    m_majorant_grid = cells * (scale * m_majorant_factor);
    dr::eval(m_majorant_grid);
    m_majorant_grid_res = res;
    m_majorant_to_local = volume->to_local();
}

MI_VARIANT Float
Medium<Float, Spectrum>::majorant_grid_eval(const Point3f &p,
                                            Mask active) const {
    Point3f pl = m_majorant_to_local * p;
    Vector3f g = pl * Vector3f(ScalarVector3f(m_majorant_grid_res));
    Vector3i cell =
        dr::clip(Vector3i(dr::floor(g)), 0,
                 Vector3i(ScalarVector3i(m_majorant_grid_res)) - 1);
    UInt32 idx = (UInt32(cell.z()) * m_majorant_grid_res.y() +
                  UInt32(cell.y())) * m_majorant_grid_res.x() +
                 UInt32(cell.x());
    return dr::gather<Float>(m_majorant_grid, idx, active);
}

MI_VARIANT std::tuple<Float, Float, typename Medium<Float, Spectrum>::Mask>
Medium<Float, Spectrum>::sample_interaction_dda(const Ray3f &ray, Float mint,
                                                Float maxt, Float sample,
                                                Mask active) const {
    ScalarVector3f res_f(m_majorant_grid_res);
    ScalarVector3i res_i(m_majorant_grid_res);
    uint32_t rx = m_majorant_grid_res.x(), ry = m_majorant_grid_res.y();

    // Reparameterize the ray segment [mint, maxt] into supergrid cell space
    Point3f  o_g = (m_majorant_to_local * ray(mint)) * Vector3f(res_f);
    Vector3f d_g = (m_majorant_to_local * ray.d) * Vector3f(res_f);

    Float t_end      = maxt - mint;
    Float tau_target = -dr::log(1.f - sample);

    Vector3i step    = dr::select(d_g >= 0.f, 1, -1);
    Vector3f t_delta = dr::rcp(dr::abs(d_g)); // world-t per cell crossing
    Vector3f next_b  = Vector3f(dr::clip(Vector3i(dr::floor(o_g)), 0,
                                         res_i - 1) +
                                dr::select(d_g >= 0.f, Vector3i(1),
                                           Vector3i(0)));
    Vector3f t_max   = dr::select(d_g != 0.f, (next_b - o_g) / d_g,
                                  dr::Infinity<Float>);

    struct DDAState {
        Mask active;
        Vector3i cell;
        Vector3f t_max;
        Float t_cur;
        Float tau_acc;
        Float t_hit;
        Float majorant;

        DRJIT_STRUCT(DDAState, active, cell, t_max, t_cur, tau_acc, t_hit,                      majorant)
    } ls = {
        active && (t_end > 0.f),
        dr::clip(Vector3i(dr::floor(o_g)), 0, res_i - 1),
        t_max,
        Float(0.f),
        Float(0.f),
        dr::Infinity<Float>,
        Float(0.f)
    };

    dr::tie(ls) = dr::while_loop(dr::make_tuple(ls),
        [](const DDAState &ls) { return ls.active; },
        [this, rx, ry, res_i, t_delta, step, t_end,
         tau_target](DDAState &ls) {
            UInt32 idx = (UInt32(ls.cell.z()) * ry + UInt32(ls.cell.y())) *
                             rx + UInt32(ls.cell.x());
            Float sigma = dr::gather<Float>(m_majorant_grid, idx, ls.active);

            Float t_next = dr::minimum(dr::min(ls.t_max), t_end);
            Float dtau   = sigma * (t_next - ls.t_cur);

            // Sample lands inside the current cell?
            Mask hit = ls.active && (sigma > 0.f) &&
                       (ls.tau_acc + dtau >= tau_target);
            dr::masked(ls.t_hit, hit) =
                ls.t_cur + (tau_target - ls.tau_acc) / sigma;
            dr::masked(ls.majorant, hit) = sigma;
            ls.active &= !hit;

            // Otherwise, accumulate and advance to the neighboring cell
            dr::masked(ls.tau_acc, ls.active) = ls.tau_acc + dtau;
            dr::masked(ls.t_cur, ls.active)   = t_next;
            ls.active &= t_next < t_end;

            Mask ax_x = ls.active && (ls.t_max.x() <= ls.t_max.y()) &&
                        (ls.t_max.x() <= ls.t_max.z());
            Mask ax_y = ls.active && !ax_x && (ls.t_max.y() <= ls.t_max.z());
            Mask ax_z = ls.active && !ax_x && !ax_y;

            dr::masked(ls.cell.x(), ax_x)  = ls.cell.x() + step.x();
            dr::masked(ls.cell.y(), ax_y)  = ls.cell.y() + step.y();
            dr::masked(ls.cell.z(), ax_z)  = ls.cell.z() + step.z();
            dr::masked(ls.t_max.x(), ax_x) = ls.t_max.x() + t_delta.x();
            dr::masked(ls.t_max.y(), ax_y) = ls.t_max.y() + t_delta.y();
            dr::masked(ls.t_max.z(), ax_z) = ls.t_max.z() + t_delta.z();

            // Leaving the supergrid also terminates the walk
            ls.active &= dr::all((ls.cell >= 0) && (ls.cell < res_i));
        });

    Mask valid = active && (ls.majorant > 0.f);
    return { mint + ls.t_hit, ls.majorant, valid };
}

MI_VARIANT
std::pair<typename Medium<Float, Spectrum>::UnpolarizedSpectrum,
          typename Medium<Float, Spectrum>::UnpolarizedSpectrum>
Medium<Float, Spectrum>::transmittance_eval_pdf(const MediumInteraction3f &mi,
                                                const SurfaceInteraction3f &si,
                                                Mask active) const {
    MI_MASKED_FUNCTION(ProfilerPhase::MediumEvaluate, active);

    Float t      = dr::minimum(mi.t, si.t) - mi.mint;
    UnpolarizedSpectrum tr  = dr::exp(-t * mi.combined_extinction);
    UnpolarizedSpectrum pdf = dr::select(si.t < mi.t, tr, tr * mi.combined_extinction);
    return { tr, pdf };
}

MI_VARIANT
typename Medium<Float, Spectrum>::UnpolarizedSpectrum
Medium<Float, Spectrum>::get_albedo(const MediumInteraction3f &mi,
                                    Mask active) const {
    MI_MASKED_FUNCTION(ProfilerPhase::MediumEvaluate, active);

    auto [sigma_s, sigma_n, sigma_t] = get_scattering_coefficients(mi, active);
    return dr::select(sigma_t > 0.f, sigma_s / sigma_t, 0.f);
}

MI_IMPLEMENT_TRAVERSE_CB(Medium, Object)
MI_INSTANTIATE_CLASS(Medium)
NAMESPACE_END(mitsuba)

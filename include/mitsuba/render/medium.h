#pragma once

#include <mitsuba/core/object.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/traits.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/render/fwd.h>
#include <drjit/call.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class MI_EXPORT_LIB Medium : public JitObject<Medium<Float, Spectrum>> {
public:
    MI_IMPORT_TYPES(PhaseFunction, Sampler, Scene, Texture, Volume);

    /// Destructor
    ~Medium();

    /// Intersects a ray with the medium's bounding box
    virtual std::tuple<Mask, Float, Float>
    intersect_aabb(const Ray3f &ray) const = 0;

    /// Returns the medium's majorant used for delta tracking
    virtual UnpolarizedSpectrum
    get_majorant(const MediumInteraction3f &mi,
                 Mask active = true) const = 0;

    /// Returns the medium coefficients Sigma_s, Sigma_n and Sigma_t evaluated
    /// at a given MediumInteraction mi
    virtual std::tuple<UnpolarizedSpectrum, UnpolarizedSpectrum,
                       UnpolarizedSpectrum>
    get_scattering_coefficients(const MediumInteraction3f &mi,
                                Mask active = true) const = 0;

    /**
     * \brief Returns the single-scattering albedo (Sigma_s / Sigma_t)
     * evaluated at a given MediumInteraction mi
     *
     * The default implementation computes the ratio from
     * \ref get_scattering_coefficients(). Media that store the albedo as an
     * independent quantity (e.g. \c homogeneous and \c heterogeneous)
     * override this with a direct lookup, which is cheaper and, when
     * differentiated, attaches the result directly to the albedo parameter
     * without spurious extinction terms in the AD graph.
     */
    virtual UnpolarizedSpectrum
    get_albedo(const MediumInteraction3f &mi, Mask active = true) const;

    /**
     * \brief Sample a free-flight distance in the medium.
     *
     * This function samples a (tentative) free-flight distance according to an
     * exponential transmittance. It is then up to the integrator to then decide
     * whether the MediumInteraction corresponds to a real or null scattering
     * event.
     *
     * \param ray      Ray, along which a distance should be sampled
     * \param sample   A uniformly distributed random sample
     * \param channel  The channel according to which we will sample the
     * free-flight distance. This argument is only used when rendering in RGB
     * modes.
     *
     * \return         This method returns a MediumInteraction.
     *                 The MediumInteraction will always be valid,
     *                 except if the ray missed the Medium's bounding box.
     */
    MediumInteraction3f sample_interaction(const Ray3f &ray, Float sample,
                                           UInt32 channel, Mask active) const;

    /**
     * \brief Compute the transmittance and PDF
     *
     * This function evaluates the transmittance and PDF of sampling a certain
     * free-flight distance The returned PDF takes into account if a medium
     * interaction occurred (mi.t <= si.t) or the ray left the medium (mi.t >
     * si.t)
     *
     * The evaluated PDF is spectrally varying. This allows to account for the
     * fact that the free-flight distance sampling distribution can depend on
     * the wavelength.
     *
     * \return   This method returns a pair of (Transmittance, PDF).
     *
     */
    std::pair<UnpolarizedSpectrum, UnpolarizedSpectrum>
    transmittance_eval_pdf(const MediumInteraction3f &mi,
                           const SurfaceInteraction3f &si,
                           Mask active) const;

    /// Return the phase function of this medium
    MI_INLINE const PhaseFunction *phase_function() const {
        return m_phase_function.get();
    }

    /// Returns whether this specific medium instance uses emitter sampling
    MI_INLINE bool use_emitter_sampling() const { return m_sample_emitters; }

    /// Returns whether this medium is homogeneous
    MI_INLINE bool is_homogeneous() const { return m_is_homogeneous; }

    /// Returns whether this medium has a spectrally varying extinction
    MI_INLINE bool has_spectral_extinction() const {
        return m_has_spectral_extinction;
    }

    /// Returns whether a majorant supergrid is available for delta tracking
    MI_INLINE bool has_majorant_grid() const {
        return m_majorant_grid_res.x() > 0;
    }

    void traverse(TraversalCallback *callback) override;

    /// Return a human-readable representation of the Medium
    std::string to_string() const override = 0;

    MI_DECLARE_PLUGIN_BASE_CLASS(Medium)

protected:
    Medium();
    Medium(const Properties &props);

    /**
     * \brief (Re-)build the majorant supergrid from the given extinction
     * volume, scaled by \c scale and inflated by \c m_majorant_factor.
     *
     * The supergrid is *derived* data (like a BVH): it is recomputed in
     * \c parameters_changed() and detached from the AD graph. A no-op when
     * \c majorant_resolution_factor is zero.
     */
    void update_majorant_grid(const Volume *volume, ScalarFloat scale);

    /// Conservative per-cell majorant lookup (nearest cell)
    Float majorant_grid_eval(const Point3f &p, Mask active) const;

    /**
     * \brief Sample a free-flight distance against the piecewise-constant
     * majorant supergrid using a DDA traversal.
     *
     * Returns the sampled distance along the ray, the local majorant of the
     * cell containing the sample, and a validity mask (false = the target
     * optical depth was not reached before \c maxt, i.e. the ray escaped).
     */
    std::tuple<Float, Float, Mask>
    sample_interaction_dda(const Ray3f &ray, Float mint, Float maxt,
                           Float sample, Mask active) const;

protected:
    ref<PhaseFunction> m_phase_function;
    bool m_sample_emitters;
    bool m_is_homogeneous;
    bool m_has_spectral_extinction;

    /// Majorant supergrid resolution divisor (0 = disabled, global majorant)
    uint32_t m_majorant_resolution_factor;
    /// Safety factor applied on top of the per-cell maxima
    ScalarFloat m_majorant_factor;
    /// Derived majorant supergrid (x-fastest layout), detached
    DynamicBuffer<Float> m_majorant_grid;
    /// Supergrid resolution; (0, 0, 0) when disabled
    ScalarVector3u m_majorant_grid_res;
    /// World-to-local transform of the extinction volume
    ScalarAffineTransform4f m_majorant_to_local;

    MI_DECLARE_TRAVERSE_CB(m_phase_function, m_majorant_grid)
};

MI_EXTERN_CLASS(Medium)
NAMESPACE_END(mitsuba)

// -----------------------------------------------------------------------
//! @{ \name Enables vectorized method calls on Dr.Jit medium arrays
// -----------------------------------------------------------------------

DRJIT_CALL_TEMPLATE_BEGIN(mitsuba::Medium)
    DRJIT_CALL_GETTER(phase_function)
    DRJIT_CALL_GETTER(use_emitter_sampling)
    DRJIT_CALL_GETTER(is_homogeneous)
    DRJIT_CALL_GETTER(has_spectral_extinction)
    DRJIT_CALL_METHOD(get_majorant)
    DRJIT_CALL_METHOD(get_albedo)
    DRJIT_CALL_METHOD(intersect_aabb)
    DRJIT_CALL_METHOD(sample_interaction)
    DRJIT_CALL_METHOD(transmittance_eval_pdf)
    DRJIT_CALL_METHOD(get_scattering_coefficients)
DRJIT_CALL_END()

//! @}
// -----------------------------------------------------------------------

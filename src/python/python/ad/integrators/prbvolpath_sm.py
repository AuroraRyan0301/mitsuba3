from __future__ import annotations # Delayed parsing of type annotations

import struct

import drjit as dr
import mitsuba as mi

from .common import mis_weight
from .prbvolpath import PRBVolpathIntegrator, index_spectrum


class _SuffixState:
    """
    Plain container used to re-enter :py:meth:`PRBVolpathSMIntegrator.sample`
    in primal mode, continuing a (detached) suffix path from a gradient probe
    location inside a medium.
    """
    def __init__(self, depth, medium, last_scatter_event,
                 last_scatter_direction_pdf, channel, active):
        self.depth = depth
        self.medium = medium
        self.last_scatter_event = last_scatter_event
        self.last_scatter_direction_pdf = last_scatter_direction_pdf
        self.channel = channel
        self.active = active


class PRBVolpathSMIntegrator(PRBVolpathIntegrator):
    r"""
    .. _integrator-prbvolpath_sm:

    Sample-Matching PRB Volumetric Integrator (:monosp:`prbvolpath_sm`)
    -------------------------------------------------------------------

    .. pluginparameters::

     * - max_depth
       - |int|
       - Specifies the longest path depth in the generated output image (where -1
         corresponds to :math:`\infty`). (Default: 6)

     * - rr_depth
       - |int|
       - Specifies the path depth, at which the implementation will begin to use
         the *russian roulette* path termination criterion. (Default: 5)

     * - hide_emitters
       - |bool|
       - Hide directly visible emitters. (Default: no, i.e. |false|)

     * - probes_per_segment
       - |int|
       - Number of gradient probe locations placed on each completed path
         segment inside a medium (the parameter :math:`\Lambda` in the paper).
         The first probe estimates direct *and* indirect in-scattered radiance
         (the latter via one recursive suffix path shared by the whole
         segment); additional probes only estimate direct lighting.
         (Default: 1; the paper uses 4)

     * - use_probe_mis
       - |bool|
       - Combine the vertex-side (free-flight) and probe-side (uniform)
         estimators of the albedo-weighted in-scattering derivative using the
         power heuristic. (Default: |true|)

     * - linear_cost
       - |bool|
       - Reservoir-based segment subsampling: the (expensive) recursive
         suffix path that estimates indirect in-scattered radiance is traced
         only *once per path*, at a segment selected with probability
         proportional to the path throughput, instead of once per segment.
         This reduces the adjoint cost from :math:`O(n^2)` to :math:`O(n)` in
         the path length, at the price of somewhat higher gradient variance.
         Direct-light probes and the matched transmittance term are still
         evaluated on every segment. (Default: |false|)

    This integrator extends the volumetric Path Replay Backpropagation
    integrator (:monosp:`prbvolpath`) with **sample matching** for extinction
    (:monosp:`sigma_t`) gradients:

    Differentiating volumetric transport with respect to the extinction
    coefficient yields two contributions with opposite signs — a *scattering*
    term (a density increase scatters more light toward the camera) and a
    *transmittance* term (a density increase attenuates all light passing
    through). Conventional estimators (e.g. differentiable delta tracking as
    used by :monosp:`prbvolpath`) evaluate the two terms at *different*
    locations, leaving their negative correlation unexploited. This integrator
    instead evaluates both terms at **shared, uniformly-sampled probe
    locations** on each path segment, activating the negative covariance and
    substantially reducing extinction-gradient variance without introducing
    bias.

    In primal mode this integrator behaves exactly like :monosp:`prbvolpath`.
    All sample-matching machinery only runs in the adjoint (backward) pass.

    Properties (inherited from :monosp:`prbvolpath`):

    - Emitter sampling (NEE), Russian roulette, surfaces + multiple media.
    - No projective sampling: geometric parameters (e.g. vertex positions)
      receive incorrect gradients.
    - Detached sampling: parameters of ideal specular objects cannot be
      optimized.
    - Forward-mode differentiation is not supported.

    See "Sample Matching for Joint Extinction Gradient Estimation in
    Differentiable Volume Rendering" (Yu et al., ACM TOG 45(4), 2026,
    https://doi.org/10.1145/3811329) for details, and
    :cite:`Vicini2021` for the underlying PRB framework.

    .. warning::
        This integrator is not supported in variants which track polarization
        states.

    .. tabs::

        .. code-tab:: python

            'type': 'prbvolpath_sm',
            'max_depth': 8,
            'probes_per_segment': 4
    """
    def __init__(self, props):
        super().__init__(props)
        self.probes_per_segment = props.get('probes_per_segment', 1)
        self.use_probe_mis = props.get('use_probe_mis', True)
        self.linear_cost = props.get('linear_cost', False)
        # Two-stage adjoint: the path-replay loop only appends compact
        # per-segment records; probe lighting estimation runs afterwards in
        # a small, coherent kernel of its own. This keeps ray-tracing calls
        # (and their register-state cost) out of the large replay kernel.
        self.defer_probes = props.get('defer_probes', True)
        # Segment-record capacity, as a multiple of the wavefront size.
        # Records past the capacity are dropped (with a warning); the default
        # is far above typical volumetric path lengths.
        self.defer_capacity = props.get('defer_capacity', 16)
        # K-slot striped segment reservoir (0 = record every segment).
        # Per lane, segment k goes to slot (k mod K); each slot runs a
        # single weighted reservoir. Bounded memory/cost: at most K segments
        # per path receive probes + a recursive suffix, with RIS compensation
        # V_slot / v_sel. Exact (all segments kept) for paths with <= K
        # segments. Sits between the quadratic and linear variants.
        self.segment_reservoir = props.get('segment_reservoir', 0)
        # Consume null collisions in a small nested loop instead of one outer
        # (full-body) iteration per collision. Mechanism test for the
        # accept-until-real design; estimator-equivalent (detached weights
        # accumulated per hop).
        self.null_inner_loop = props.get('null_inner_loop', False)
        # Same mechanism, implemented as a C++ walk inside the Medium
        # (Medium::sample_real_interaction): candidates never surface to the
        # Python loop at all. For the C++-vs-nested-Python comparison.
        self.real_interaction_cpp = props.get('real_interaction_cpp', False)
        # Use the fused DDA+walk C++ variant (implies real_interaction_cpp)
        self.real_interaction_fused = props.get('real_interaction_fused', True)
        if self.real_interaction_fused:
            self.real_interaction_cpp = True

        if self.probes_per_segment < 1:
            raise Exception('"probes_per_segment" must be >= 1')

    @dr.syntax
    def sample(self,
               mode: dr.ADMode,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               δL: Optional[mi.Spectrum],
               state_in: Optional[mi.Spectrum],
               active: mi.Bool,
               path_state=None,
               **kwargs # Absorbs unused arguments
    ) -> Tuple[mi.Spectrum, mi.Bool, List[mi.Float], mi.Spectrum]:
        self.prepare_scene(scene)

        if mode == dr.ADMode.Forward:
            raise RuntimeError("PRBVolpathSMIntegrator doesn't support "
                               "forward-mode differentiation!")

        is_primal = mode == dr.ADMode.Primal

        ray = mi.Ray3f(ray)
        L = mi.Spectrum(0 if is_primal else state_in) # Radiance accumulator
        δL = mi.Spectrum(δL if δL is not None else 0) # Differential/adjoint radiance
        throughput = mi.Spectrum(1)                   # Path throughput weight
        η = mi.Float(1)                               # Index of refraction
        active = mi.Bool(active)

        si = dr.zeros(mi.SurfaceInteraction3f)
        needs_intersection = mi.Bool(True)
        valid_ray = mi.Bool(False)

        if path_state is not None:
            # Re-entry: continue a detached suffix path from a probe location
            # (only used by the adjoint pass; the recursion itself is primal).
            if not is_primal:
                raise RuntimeError('Recursive suffix rays must be traced in primal mode')
            depth = mi.UInt32(path_state.depth)
            medium = mi.MediumPtr(path_state.medium)
            last_scatter_event = mi.Interaction3f(path_state.last_scatter_event)
            last_scatter_direction_pdf = mi.Float(path_state.last_scatter_direction_pdf)
            channel = mi.UInt32(path_state.channel)
            specular_chain = mi.Bool(False)
        else:
            depth = mi.UInt32(0)
            last_scatter_event = dr.zeros(mi.Interaction3f)
            last_scatter_direction_pdf = mi.Float(1.0)
            # TODO: support sensors inside media
            medium = dr.zeros(mi.MediumPtr)
            specular_chain = mi.Bool(True)
            channel = 0
            if mi.is_rgb:
                # Sample a color channel to sample free-flight distances
                n_channels = dr.size_v(mi.Spectrum)
                channel = mi.UInt32(dr.minimum(n_channels * sampler.next_1d(active), n_channels - 1))

        # Secondary sampler driving all probe/suffix estimation in the adjoint.
        # One primary sample is consumed in *both* passes so that the primary
        # random number sequence stays aligned between primal and adjoint
        # (required by path replay).
        alt_seed_f = sampler.next_1d(active)
        alt_sampler = None
        if dr.hint(not is_primal, mode='scalar'):
            alt_seed = struct.unpack('!I', struct.pack('!f', alt_seed_f[0]))[0]
            alt_sampler = sampler.fork()
            alt_sampler.seed(mi.sample_tea_32(alt_seed, 1)[0],
                             sampler.wavefront_size())
        del alt_seed_f

        # Deferred-probe record buffers (see `defer_probes` in __init__).
        # Only the top-level adjoint pass defers; recursive suffix rays and
        # the primal pass never reach the probe code.
        defer = (not is_primal) and self.defer_probes and (path_state is None)
        if dr.hint(defer, mode='scalar'):
            dfr_n = sampler.wavefront_size()
            dfr_cap = int(dfr_n) * self.defer_capacity
            dfr_cap_o = dr.opaque(mi.UInt32, dfr_cap)
            dfr_ctr = dr.zeros(mi.UInt32, 1)
            dfr = {k: dr.zeros(mi.Float, dfr_cap) for k in
                   ('ox', 'oy', 'oz', 'dx', 'dy', 'dz', 'itv',
                    'atx', 'aty', 'atz', 'asx', 'asy', 'asz', 'nu', 'nv')}
            K_res = self.segment_reservoir
            if dr.hint(K_res > 0, mode='scalar'):
                dfr_cap = int(dfr_n) * K_res
                dfr = {k: dr.zeros(mi.Float, dfr_cap) for k in dfr}
                dfr['vsl'] = dr.zeros(mi.Float, dfr_cap)   # scalar v of retained
                dfr['Vj'] = dr.zeros(mi.Float, dfr_cap)    # slot weight sum (post-loop)
                if K_res != 4:
                    raise Exception('segment_reservoir currently supports K=4')
                lane_idx = dr.arange(mi.UInt32, dfr_n)
            dfr['dep'] = dr.zeros(mi.UInt32, dfr_cap)
            dfr['ch'] = dr.zeros(mi.UInt32, dfr_cap)
            dfr['med'] = dr.zeros(mi.MediumPtr, dfr_cap)

        # Sample-matching segment state: a "segment" spans from the last real
        # scatter vertex (or last surface interaction) to the next one. Null
        # interactions do not end a segment: the direction is unchanged, so we
        # accumulate the distance traveled across them in `seg_dist`.
        seg_origin = mi.Point3f(ray.o)
        seg_dist = mi.Float(0.0)

        # Reservoir for the linear-cost variant: selects one segment (with
        # probability proportional to the path throughput) whose main probe
        # will receive the single, deferred recursive suffix path estimating
        # indirect in-scattered radiance. Weighted reservoir sampling with
        # RIS-style reweighting keeps the estimator unbiased.
        res_wsum = mi.Float(0.0)         # sum of scalar reservoir weights seen
        res_w = mi.Spectrum(0.0)         # spectral throughput of the retained sample
        res_v = mi.Float(0.0)            # scalar reservoir weight of that sample
        res_mei = dr.zeros(mi.MediumInteraction3f)  # retained probe location
        res_depth = mi.UInt32(0)         # suffix entry depth at that probe
        res_interval = mi.Float(0.0)     # retained segment length (inv. pdf)
        res_active = mi.Bool(False)      # reservoir holds a valid sample
        # K=4 striped segment reservoir (see `segment_reservoir`): loop-state
        # variables must be unconditionally initialized at function scope for
        # @dr.syntax to thread them through the symbolic loop.
        seg_ctr = mi.UInt32(0)
        res_V0 = mi.Float(0.0); res_V1 = mi.Float(0.0)
        res_V2 = mi.Float(0.0); res_V3 = mi.Float(0.0)

        while dr.hint(active,
                      label=f"PRB Sample Matching ({mode.name})"):
            active &= dr.any(throughput != 0.0)

            #--------------------- Perform russian roulette --------------------

            q = dr.minimum(dr.max(throughput) * dr.square(η), 0.99)
            perform_rr = (depth > self.rr_depth)
            active &= (sampler.next_1d(active) < q) | ~perform_rr
            throughput[perform_rr] = throughput * dr.rcp(q)

            active_medium = active & (medium != None)
            active_surface = active & ~active_medium

            with dr.resume_grad(when=not is_primal):
                #--------------------- Sample medium interaction -------------------

                # Handle medium sampling and potential medium escape
                if dr.hint(self.real_interaction_cpp and
                           self.handle_null_scattering, mode='scalar'):
                    # C++ walk: null collisions are consumed inside
                    # Medium::sample_real_interaction; the loop body only
                    # ever sees real scatters or escapes.
                    intersect = needs_intersection & active_medium
                    si[intersect] = scene.ray_intersect(ray, intersect)
                    needs_intersection &= ~active_medium
                    seed32 = mi.UInt32(sampler.next_1d(active_medium)
                                       * 4294967040.0)
                    with dr.suspend_grad():
                        if dr.hint(self.real_interaction_fused, mode='scalar'):
                            mei, w_cpp, sp_cpp = \
                                medium.sample_real_interaction_fused(
                                    ray, dr.detach(si.t), seed32, channel,
                                    active_medium)
                        else:
                            mei, w_cpp, sp_cpp = medium.sample_real_interaction(
                                ray, dr.detach(si.t), seed32, channel,
                                active_medium)
                    mei.t = dr.detach(mei.t)
                else:
                    u = sampler.next_1d(active_medium)
                    mei = medium.sample_interaction(ray, u, channel, active_medium)
                    mei.t = dr.detach(mei.t)

                    ray.maxt[active_medium & medium.is_homogeneous() & mei.is_valid()] = mei.t
                    intersect = needs_intersection & active_medium
                    si[intersect] = scene.ray_intersect(ray, intersect)

                    needs_intersection &= ~active_medium
                    mei.t[active_medium & (si.t < mei.t)] = dr.inf

                inner_sp = mi.Float(1.0)
                inner_w = mi.Spectrum(1.0)   # null-hop weights (merged below)
                if dr.hint(self.null_inner_loop and self.handle_null_scattering,
                           mode='scalar'):
                    # Walk across null collisions in a tiny nested loop; the
                    # outer (fat) loop body then only ever sees real scatters
                    # or escapes. Per-hop detached weights match the outer
                    # implementation exactly.
                    walk = active_medium & mei.is_valid()
                    # The walk is fully detached (SM estimates sigma_t
                    # derivatives via probes); suspend AD so the nested loop
                    # carries no differentiable state.
                    with dr.suspend_grad():
                        while dr.hint(walk, label='Null-collision walk'):
                            sp_h = dr.detach(dr.mean(mei.sigma_t / mei.combined_extinction))
                            inner_sp[walk] = sp_h
                            nul = walk & (sampler.next_1d(walk) >= sp_h)
                            # null hop weight: (tr/pdf) * sigma_n / (1 - p)
                            tr_h, pdf_h = medium.transmittance_eval_pdf(mei, si, nul)
                            w_h = dr.select(index_spectrum(pdf_h, channel) > 0,
                                            tr_h / index_spectrum(pdf_h, channel), 0.0)
                            inner_w[nul] *= dr.detach(w_h * mei.sigma_n) / (1 - sp_h)
                            ray.o[nul] = dr.detach(mei.p)
                            si.t[nul] = si.t - dr.detach(mei.t)
                            seg_dist[nul] = seg_dist + dr.detach(mei.t)
                            u_h = sampler.next_1d(nul)
                            mei[nul] = medium.sample_interaction(ray, u_h, channel, nul)
                            mei.t = dr.detach(mei.t)
                            mei.t[nul & (si.t < mei.t)] = dr.inf
                            walk = nul & mei.is_valid()

                # Sample matching: the free-flight/transmittance ratio is used
                # *detached*. The sigma_t derivatives of the path prefix are
                # estimated by the matched segment probes below instead, where
                # the transmittance and in-scattering terms share the same
                # sample locations (activating their negative correlation).
                if dr.hint(self.real_interaction_cpp and
                           self.handle_null_scattering, mode='scalar'):
                    tr, tr_pdf = mi.Spectrum(1.0), mi.Float(1.0)  # in w_cpp
                else:
                    tr, free_flight_pdf = medium.transmittance_eval_pdf(mei, si, active_medium)
                    tr_pdf = index_spectrum(free_flight_pdf, channel)
                weight = mi.Spectrum(1.0)
                if dr.hint(self.real_interaction_cpp and
                           self.handle_null_scattering, mode='scalar'):
                    # The C++ walk already includes all hop factors
                    weight[active_medium] *= dr.detach(w_cpp)
                else:
                    weight[active_medium] *= dr.detach(dr.select(tr_pdf > 0.0, tr / tr_pdf, 0.0))
                if dr.hint(self.null_inner_loop and self.handle_null_scattering,
                           mode='scalar'):
                    # Fold in the null-hop weights accumulated by the walk
                    weight[active_medium] *= dr.detach(inner_w)

                escaped_medium = active_medium & ~mei.is_valid()
                active_medium &= mei.is_valid()

                # NEE direction sample for this bounce, shared between the
                # path vertex and all gradient probes of the segment (the
                # emitter sample is "matched" as well).
                nee_dir_sample = sampler.next_2d(active)

                # Handle null and real scatter events
                if dr.hint(self.real_interaction_cpp and
                           self.handle_null_scattering, mode='scalar'):
                    scatter_prob = sp_cpp
                    act_null_scatter = mi.Bool(False)
                    act_medium_scatter = active_medium
                elif dr.hint(self.null_inner_loop and self.handle_null_scattering,
                           mode='scalar'):
                    # Inner walk already consumed all null collisions.
                    scatter_prob = inner_sp
                    act_null_scatter = mi.Bool(False)
                    act_medium_scatter = active_medium
                elif dr.hint(self.handle_null_scattering, mode='scalar'):
                    scatter_prob = dr.detach(dr.mean(mei.sigma_t / mei.combined_extinction))
                    act_null_scatter = (sampler.next_1d(active_medium) >= scatter_prob) & active_medium
                    act_medium_scatter = ~act_null_scatter & active_medium
                    weight[act_null_scatter] *= dr.detach(mei.sigma_n) / (1 - scatter_prob)
                else:
                    scatter_prob = mi.Float(1.0)
                    act_null_scatter = mi.Bool(False)
                    act_medium_scatter = active_medium

                depth[act_medium_scatter] += 1
                last_scatter_event[act_medium_scatter] = dr.detach(mei)

                # Segment-end masks, captured *before* the depth cutoff so
                # that the final path segment still receives its probes.
                seg_end_scatter = mi.Bool(act_medium_scatter)
                seg_end = seg_end_scatter | escaped_medium

                # Don't estimate lighting if we exceeded number of bounces
                active &= depth < self.max_depth
                act_medium_scatter &= active
                if dr.hint(self.handle_null_scattering, mode='scalar'):
                    ray.o[act_null_scatter] = dr.detach(mei.p)
                    si.t[act_null_scatter] = si.t - dr.detach(mei.t)
                    seg_dist[act_null_scatter] = seg_dist + dr.detach(mei.t)

                # Path throughput *excluding* the current segment's hop weight
                # (tr/pdf = 1/majorant for a sampled interaction) and vertex
                # scattering coefficient — the adjoint weight of the probes'
                # in-scattering term, which re-evaluates sigma_t * albedo at
                # the probe location in their stead. Note: the hop weight and
                # vertex factor jointly reduce to `albedo` in expectation
                # (delta tracking); the probe estimator's derivation absorbs
                # exactly that whole product, so neither factor may appear
                # here.
                throughput_seg = mi.Spectrum(throughput)

                weight[act_medium_scatter] *= dr.detach(mei.sigma_s) / scatter_prob
                throughput *= weight  # (all factors above are detached)

                # Attached single-scattering albedo at the vertex
                # (Medium::get_albedo attaches directly to the albedo
                # parameter, without spurious extinction terms).
                albedo_v = dr.select(seg_end_scatter,
                                     medium.get_albedo(mei, seg_end_scatter),
                                     mi.Spectrum(1.0))
                mei = dr.detach(mei)

                if dr.hint(not is_primal, mode='scalar'):
                    # ==================== Sample matching ====================
                    # (Replaces prbvolpath's attached free-flight weight
                    # backpropagation.)

                    # (1) Vertex-side albedo derivative, MIS-combined with the
                    #     probe-side estimator (power heuristic, sigma_t^2).
                    if dr.hint(self.use_probe_mis and dr.grad_enabled(albedo_v), mode='scalar'):
                        s2 = dr.square(mei.sigma_t)
                        mis_v = s2 / (1 + s2)
                        Lo_alb = dr.detach(dr.select(
                            seg_end_scatter,
                            L / dr.maximum(1e-8, dr.detach(albedo_v)), 0.0))
                        dr.backward(mis_v * δL * albedo_v * Lo_alb)

                    # (2) Matched gradient probes on the completed segment:
                    #     the transmittance term (-sigma_t) and in-scattering
                    #     term (+sigma_t * albedo * Li) are evaluated at
                    #     shared, uniformly-sampled locations.
                    interval = dr.detach(dr.select(
                        seg_end,
                        seg_dist + dr.select(escaped_medium, si.t, mei.t),
                        0.0))
                    # Rare geometric edge case: a lane inside the medium whose
                    # forward intersection failed (si.t = inf) while the DDA
                    # exhausted the segment -> interval = inf -> probe position
                    # at infinity -> NaN gradients scattered into clamped
                    # boundary voxels. Exclude such lanes from probing.
                    seg_end &= dr.isfinite(interval)
                    interval = dr.select(seg_end, interval, 0.0)
                    suffix_depth = dr.select(escaped_medium, depth + 1, depth)
                    # Overlap of the segment with the density grid's bbox
                    # (probe domain; see _sample_segment_probes for why the
                    # domain must exclude the out-of-grid clamp region).
                    sbb_hit, sbb0, sbb1 = medium.intersect_aabb(
                        mi.Ray3f(mi.Point3f(seg_origin), mi.Vector3f(ray.d)))
                    seg_t0 = dr.detach(dr.clip(sbb0, 0.0, interval))
                    seg_sub = dr.select(
                        sbb_hit,
                        dr.detach(dr.clip(sbb1, 0.0, interval)) - seg_t0, 0.0)
                    if dr.hint(defer, mode='scalar'):
                        # Append the segment record; probe estimation happens
                        # in the compact second-stage kernel after the loop.
                        at = dr.detach(δL * L)
                        asc = dr.detach(δL * throughput_seg)
                        if dr.hint(self.segment_reservoir > 0, mode='scalar'):
                            # K-slot striped reservoir: slot = seg index mod K
                            slot_id = seg_ctr % 4
                            seg_ctr[seg_end] = seg_ctr + 1
                            v = dr.mean(dr.detach(throughput_seg))
                            u_res = alt_sampler.next_1d(seg_end)
                            m0 = seg_end & (slot_id == 0)
                            m1 = seg_end & (slot_id == 1)
                            m2 = seg_end & (slot_id == 2)
                            m3 = seg_end & (slot_id == 3)
                            res_V0 = res_V0 + dr.select(m0, v, 0.0)
                            res_V1 = res_V1 + dr.select(m1, v, 0.0)
                            res_V2 = res_V2 + dr.select(m2, v, 0.0)
                            res_V3 = res_V3 + dr.select(m3, v, 0.0)
                            Vmine = dr.select(m0, res_V0,
                                    dr.select(m1, res_V1,
                                    dr.select(m2, res_V2, res_V3)))
                            r = dr.select(Vmine > 0, v / Vmine, 0.0)
                            change = seg_end & (u_res <= r)
                            slot = lane_idx * 4 + slot_id
                            ok = change
                            dr.scatter(dfr['vsl'], v, slot, ok)
                        else:
                            slot = dr.scatter_inc(dfr_ctr, mi.UInt32(0), seg_end)
                            ok = seg_end & (slot < dfr_cap_o)
                        for _k, _v in (('ox', seg_origin.x), ('oy', seg_origin.y),
                                       ('oz', seg_origin.z), ('dx', ray.d.x),
                                       ('dy', ray.d.y), ('dz', ray.d.z),
                                       ('itv', interval),
                                       ('atx', at.x), ('aty', at.y), ('atz', at.z),
                                       ('asx', asc.x), ('asy', asc.y), ('asz', asc.z),
                                       ('nu', nee_dir_sample.x), ('nv', nee_dir_sample.y)):
                            dr.scatter(dfr[_k], dr.detach(_v), slot, ok)
                        dr.scatter(dfr['dep'], suffix_depth, slot, ok)
                        dr.scatter(dfr['ch'], channel, slot, ok)
                        dr.scatter(dfr['med'], medium, slot, ok)
                        # Reservoir candidate for the deferred indirect suffix
                        # (linear variant): only the *location* is needed here.
                        # Sampled from the bbox-clipped probe domain (matches
                        # _sample_segment_probes).
                        mei_main = mi.MediumInteraction3f(mei)
                        mei_main.t = dr.fma(alt_sampler.next_1d(seg_end),
                                            seg_sub, seg_t0)
                        mei_main.p = dr.fma(mi.Vector3f(ray.d), mei_main.t,
                                            mi.Point3f(seg_origin))
                    else:
                        mei_main = self._sample_segment_probes(
                            scene, medium, channel, alt_sampler, mei,
                            mi.Point3f(seg_origin), mi.Vector3f(ray.d), interval,
                            δL * L,               # adjoint of the transmittance term
                            δL * throughput_seg,  # adjoint of the in-scattering term
                            nee_dir_sample, suffix_depth, seg_end,
                            include_indirect=not self.linear_cost)

                    # (3) Linear-cost variant: instead of tracing a recursive
                    #     suffix on every segment, retain one segment's main
                    #     probe in a throughput-weighted reservoir; the
                    #     deferred suffix is traced once, after the loop.
                    if dr.hint(self.linear_cost, mode='scalar'):
                        # Reservoir with a *scalar* selection weight
                        # (v = mean(w), additive, so P(keep i) = v_i/V exactly)
                        # and *spectral* compensation w * (V / v_sel): the
                        # per-channel expectation matches the quadratic
                        # estimator exactly, even for strongly colored
                        # throughput. Using mean(w/wsum) as the selection
                        # probability while compensating per channel (as in
                        # the original DRT reservoir) leaves a second-order
                        # color bias.
                        w = dr.select(seg_end, dr.detach(throughput_seg), 0.0)
                        v = dr.mean(w)
                        res_wsum += v
                        ratio = dr.select(res_wsum > 0, v / res_wsum, 0.0)
                        change = seg_end & (alt_sampler.next_1d(seg_end) <= ratio)
                        res_w[change] = w
                        res_v[change] = v
                        res_mei[change] = mei_main
                        res_depth[change] = suffix_depth
                        # Inverse pdf of the suffix location sample: the
                        # bbox-clipped probe-domain length, NOT the full
                        # segment length.
                        res_interval[change] = seg_sub
                        res_active |= change
                    # =========================================================

                phase_ctx = mi.PhaseFunctionContext(sampler)
                phase = mei.medium.phase_function()
                phase[~act_medium_scatter] = dr.zeros(mi.PhaseFunctionPtr)

                #--------------------- Surface Interactions --------------------

                active_surface |= escaped_medium
                intersect = active_surface & needs_intersection
                si[intersect] = scene.ray_intersect(ray, intersect)

                # ---------------------- Hide area emitters ----------------------

                if dr.hint(self.hide_emitters, mode='scalar'):
                    # Are we on the first segment and did we hit an area emitter?
                    # If so, skip all area emitters along this ray
                    skip_emitters = (
                        si.is_valid() &
                        (si.shape.emitter() != None) &
                        (depth == 0) &
                        intersect
                    )

                    ray_skip = si.spawn_ray(ray.d)
                    pi = self.skip_area_emitters(scene, ray_skip, True, skip_emitters)
                    si_after_skip = pi.compute_surface_interaction(ray, mi.RayFlags.All, skip_emitters)
                    si[skip_emitters] = si_after_skip

                # ----------------- Intersection with emitters -----------------

                ray_from_camera = active_surface & (depth == 0)
                count_direct = ray_from_camera | specular_chain
                emitter = si.emitter(scene)
                active_e = active_surface & (emitter != None) & ~((depth == 0) & self.hide_emitters)

                # Get the PDF of sampling this emitter using next event estimation
                ds = mi.DirectionSample3f(scene, si, last_scatter_event)
                if dr.hint(self.use_nee, mode='scalar'):
                    emitter_pdf = scene.pdf_emitter_direction(last_scatter_event, ds, active_e)
                else:
                    emitter_pdf = 0.0
                emitted = emitter.eval(si, active_e)
                contrib = dr.select(count_direct, throughput * emitted,
                                    throughput * mis_weight(last_scatter_direction_pdf, emitter_pdf) * emitted)
                L[active_e] += dr.detach(contrib if is_primal else -contrib)
                if dr.hint(not is_primal and dr.grad_enabled(contrib), mode='scalar'):
                    dr.backward(δL * contrib)

                active_surface &= si.is_valid()
                ctx = mi.BSDFContext()
                bsdf = si.bsdf(ray)

                # ---------------------- Emitter sampling ----------------------

                if dr.hint(self.use_nee, mode='scalar'):
                    active_e_surface = active_surface & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth) & (depth + 1 < self.max_depth)
                    sample_emitters = mei.medium.use_emitter_sampling()
                    specular_chain &= ~act_medium_scatter
                    specular_chain |= act_medium_scatter & ~sample_emitters

                    active_e_medium = act_medium_scatter & sample_emitters
                    active_e = active_e_surface | active_e_medium

                    nee_sampler = sampler if is_primal else sampler.clone()
                    emitted, ds = self.sample_emitter(mei, si, active_e_medium, active_e_surface,
                        scene, sampler, medium, channel, active_e, mode=dr.ADMode.Primal,
                        dir_sample=nee_dir_sample)

                    # Query the BSDF for that emitter-sampled direction
                    bsdf_val, bsdf_pdf = bsdf.eval_pdf(ctx, si, si.to_local(ds.d), active_e_surface)
                    phase_val, phase_pdf = phase.eval_pdf(phase_ctx, mei, ds.d, active_e_medium)
                    nee_weight = dr.select(active_e_surface, bsdf_val, phase_val)
                    nee_directional_pdf = dr.select(ds.delta, 0.0, dr.select(active_e_surface, bsdf_pdf, phase_pdf))

                    contrib = throughput * nee_weight * mis_weight(ds.pdf, nee_directional_pdf) * emitted
                    L[active_e] += dr.detach(contrib if is_primal else -contrib)

                    if dr.hint(not is_primal, mode='scalar'):
                        self.sample_emitter(mei, si, active_e_medium, active_e_surface,
                            scene, nee_sampler, medium, channel, active_e, adj_emitted=contrib,
                            δL=δL, mode=mode, dir_sample=nee_dir_sample)

                        if dr.hint(dr.grad_enabled(nee_weight) or dr.grad_enabled(emitted), mode='scalar'):
                            dr.backward(δL * contrib)

                #-------------------- Phase function sampling ------------------

                valid_ray |= act_medium_scatter
                with dr.suspend_grad():
                    wo, phase_weight, phase_pdf = phase.sample(phase_ctx, mei,
                                                               sampler.next_1d(act_medium_scatter),
                                                               sampler.next_2d(act_medium_scatter),
                                                               act_medium_scatter)
                act_medium_scatter &= phase_pdf > 0.0

                # Re evaluate the phase function value in an attached manner
                phase_eval, _ = phase.eval_pdf(phase_ctx, mei, wo, act_medium_scatter)
                if dr.hint(not is_primal and dr.grad_enabled(phase_eval), mode='scalar'):
                    Lo = phase_eval * dr.detach(dr.select(act_medium_scatter, L / dr.maximum(1e-8, phase_eval), 0.0))
                    if mode == dr.ADMode.Backward:
                        dr.backward_from(δL * Lo)
                    else:
                        δL += dr.forward_to(Lo)

                throughput[act_medium_scatter] *= phase_weight
                ray[act_medium_scatter] = mei.spawn_ray(wo)
                needs_intersection |= act_medium_scatter
                last_scatter_direction_pdf[act_medium_scatter] = phase_pdf

                # ------------------------ BSDF sampling -----------------------

                with dr.suspend_grad():
                    bs, bsdf_weight = bsdf.sample(ctx, si,
                                                  sampler.next_1d(active_surface),
                                                  sampler.next_2d(active_surface),
                                                  active_surface)
                    active_surface &= bs.pdf > 0

                bsdf_eval = bsdf.eval(ctx, si, bs.wo, active_surface)

                if dr.hint(not is_primal and dr.grad_enabled(bsdf_eval), mode='scalar'):
                    Lo = bsdf_eval * dr.detach(dr.select(active_surface, L / dr.maximum(1e-8, bsdf_eval), 0.0))
                    if dr.hint(mode == dr.ADMode.Backward, mode='scalar'):
                        dr.backward_from(δL * Lo)
                    else:
                        δL += dr.forward_to(Lo)

                throughput[active_surface] *= bsdf_weight
                η[active_surface] *= bs.eta
                bsdf_ray = si.spawn_ray(si.to_world(bs.wo))
                ray[active_surface] = bsdf_ray

                needs_intersection |= active_surface
                non_null_bsdf = active_surface & ~mi.has_flag(bs.sampled_type, mi.BSDFFlags.Null)
                depth[non_null_bsdf] += 1

                # update the last scatter PDF event if we encountered a non-null scatter event
                last_scatter_event[non_null_bsdf] = si
                last_scatter_direction_pdf[non_null_bsdf] = bs.pdf

                valid_ray |= non_null_bsdf
                specular_chain |= non_null_bsdf & mi.has_flag(bs.sampled_type, mi.BSDFFlags.Delta)
                specular_chain &= ~(active_surface & mi.has_flag(bs.sampled_type, mi.BSDFFlags.Smooth))
                has_medium_trans = active_surface & si.is_medium_transition()
                medium[has_medium_trans] = si.target_medium(ray.d)

                # A new segment starts at every real scatter vertex and at
                # every surface interaction (incl. null boundary crossings).
                seg_origin[act_medium_scatter] = dr.detach(mei.p)
                seg_origin[active_surface] = dr.detach(si.p)
                seg_dist[act_medium_scatter | active_surface] = 0.0

                active &= (active_surface | active_medium)

        # ---- Deferred-probe pass: estimate probe lighting in a compact,
        # coherent kernel of its own (records were appended in the loop). ----
        if dr.hint(defer, mode='scalar'):
            if dr.hint(self.segment_reservoir > 0, mode='scalar'):
                for _j, _V in ((0, res_V0), (1, res_V1),
                               (2, res_V2), (3, res_V3)):
                    dr.scatter(dfr['Vj'], _V, lane_idx * 4 + _j)
                self._flush_deferred_probes(scene, dfr, None, dfr_cap,
                                            reservoir=True)
            else:
                self._flush_deferred_probes(scene, dfr, dfr_ctr, dfr_cap)

        # ---- Linear-cost variant: deferred indirect in-scattering probe ----
        # Trace the single recursive suffix path at the reservoir-selected
        # probe and deposit the indirect part of the matched in-scattering
        # derivative. The RIS weight `wsum * w / mean(w)` replaces the sum of
        # per-segment `throughput_seg` adjoints, keeping the estimator
        # unbiased (the transmittance and direct-light terms were already
        # deposited per segment).
        if dr.hint(not is_primal and self.linear_cost, mode='scalar'):
            w_out = dr.select(res_v > 0, res_wsum * res_w / res_v, 0.0)
            fin_active = res_active & (res_depth < self.max_depth)

            phase_ctx = mi.PhaseFunctionContext(alt_sampler)
            phase = res_mei.medium.phase_function()
            phase[~fin_active] = dr.zeros(mi.PhaseFunctionPtr)

            with dr.suspend_grad():
                ind_Li = self._probe_indirect(
                    scene, channel, alt_sampler, res_mei, phase_ctx, phase,
                    res_depth, fin_active)
            # Evaluate the loop outputs feeding the backward below before
            # traversing the AD graph: on the LLVM backend, running the eager
            # backward on the still-unevaluated loop-exit graph silently
            # drops part of the suffix gradient (CUDA is unaffected;
            # observed as a -35% albedo gradient deficit that disappears
            # with any forced evaluation).
            dr.eval(res_active, res_wsum, res_w, res_v, res_interval,
                    res_depth, ind_Li, δL, res_mei)

            with dr.resume_grad():
                sigma_s_r, _, sigma_t_r = \
                    res_mei.medium.get_scattering_coefficients(res_mei, res_active)
                # Attached albedo via sigma_s / sigma_t, as in
                # _sample_segment_probes (one vcall less).
                albedo_r = sigma_s_r / dr.maximum(sigma_t_r, 1e-8)
                if dr.hint(self.use_probe_mis, mode='scalar'):
                    mis_p = dr.rcp(1 + dr.square(dr.detach(sigma_t_r)))
                else:
                    mis_p = mi.Spectrum(1.0)
                contrib = (sigma_t_r * dr.detach(albedo_r)
                           + mis_p * dr.detach(sigma_t_r) * albedo_r) \
                          * (δL * w_out) * ind_Li * res_interval
                if dr.hint(dr.grad_enabled(contrib), mode='scalar'):
                    safe = res_active & dr.all(dr.isfinite(contrib))
                    # Evaluated context: keep graph edges (see
                    # _sample_segment_probes).
                    dr.backward(dr.select(safe, contrib, 0.0),
                                flags=dr.ADFlag.ClearVertices)

        return L if is_primal else δL, valid_ray, [], L

    def _flush_deferred_probes(self, scene, dfr, dfr_ctr, dfr_cap,
                               reservoir=False):
        """
        Second stage of the deferred-probe design: gather the per-segment
        records written by the path-replay loop and run the (ray-tracing
        heavy) probe estimation at segment granularity. Reading the record
        counter forces the replay kernel to execute first, so the probe work
        lands in a separate, much smaller kernel with compacted lanes.
        """
        if reservoir:
            n_total = n = dfr_cap      # fixed lane*K layout, no counter
        else:
            n_total = int(dfr_ctr[0])
            n = min(n_total, dfr_cap)
        if n_total > dfr_cap:
            mi.Log(mi.LogLevel.Warn,
                   f'prbvolpath_sm: deferred-probe buffer overflow '
                   f'({n_total} > {dfr_cap} records); increase '
                   f'"defer_capacity" to keep the estimator exact.')
        if n == 0:
            return
        idx = dr.arange(mi.UInt32, n)
        g = lambda k: dr.gather(mi.Float, dfr[k], idx)
        origin = mi.Point3f(g('ox'), g('oy'), g('oz'))
        seg_dir = mi.Vector3f(g('dx'), g('dy'), g('dz'))
        interval = g('itv')
        adj_trans = mi.Spectrum(g('atx'), g('aty'), g('atz'))
        adj_scatt = mi.Spectrum(g('asx'), g('asy'), g('asz'))
        nee_dir = mi.Point2f(g('nu'), g('nv'))
        suffix_depth = dr.gather(mi.UInt32, dfr['dep'], idx)
        channel = dr.gather(mi.UInt32, dfr['ch'], idx)
        medium = dr.gather(mi.MediumPtr, dfr['med'], idx)

        active = mi.Bool(True)
        if reservoir:
            # RIS compensation: the retained segment stands in for its slot's
            # whole group; occupied slots have v_sel > 0.
            vsl = g('vsl'); Vj = g('Vj')
            active = vsl > 0
            comp = dr.select(active, Vj / dr.maximum(vsl, 1e-30), 0.0)
            adj_trans *= comp
            adj_scatt *= comp

        smp = mi.load_dict({'type': 'independent'})
        smp.seed(dr.opaque(mi.UInt32, n ^ 0x9E3779B9), n)

        mei = dr.zeros(mi.MediumInteraction3f, n)
        mei.medium = medium
        mei.p = origin

        self._sample_segment_probes(scene, medium, channel, smp, mei,
                                    origin, seg_dir, interval,
                                    adj_trans, adj_scatt, nee_dir,
                                    suffix_depth, active,
                                    include_indirect=(reservoir or
                                                      not self.linear_cost))

    def _sample_segment_probes(self, scene, medium, channel, alt_sampler, mei,
                               seg_origin, seg_dir, interval, adj_trans, adj_scatt,
                               nee_dir_sample, suffix_depth, active,
                               include_indirect=True):
        """
        Sample-matched gradient probes for one completed path segment
        (adjoint pass only).

        Places `probes_per_segment` locations uniformly on the segment
        `seg_origin + t * seg_dir, t in [0, interval]` and deposits, at each
        probe location `y`:

          - the transmittance derivative  `-sigma_t(y) * adj_trans`, and
          - the in-scattering derivative  `+d(sigma_t*albedo)(y) * adj_scatt * Li(y)`,

        where both terms share the *same* `sigma_t(y)` evaluation — this is
        the sample-matching estimator that activates the negative correlation
        between the two terms. `Li(y)` is decomposed into direct lighting
        (estimated at every probe with a shared NEE direction sample) and
        indirect lighting (estimated with a single recursive suffix path,
        shared by the whole segment).
        """
        n_probes = self.probes_per_segment
        within = active & (suffix_depth < self.max_depth)

        # Restrict the probe domain to the segment's overlap with the density
        # grid's bounding box. Transport treats the region outside that box as
        # vacuum (free flight), so the density derivative vanishes there —
        # but Volume::eval() *clamps* lookup coordinates, so probing outside
        # the box would deposit spurious gradients into the boundary voxels
        # whenever the medium's shape extends beyond the grid.
        seg_ray = mi.Ray3f(mi.Point3f(seg_origin), mi.Vector3f(seg_dir))
        bb_hit, bb0, bb1 = medium.intersect_aabb(seg_ray)
        t0 = dr.detach(dr.clip(bb0, 0.0, interval))
        t1 = dr.detach(dr.clip(bb1, 0.0, interval))
        sub_len = dr.select(active & bb_hit, t1 - t0, 0.0)
        active = active & (sub_len > 0)
        within &= active

        # Probe interactions inherit the frame/wavelengths/medium pointer of
        # the segment's medium interaction; only position/distance change.
        mei_sub = mi.MediumInteraction3f(mei)

        phase_ctx = mi.PhaseFunctionContext(alt_sampler)
        phase = mei_sub.medium.phase_function()
        phase[~active] = dr.zeros(mi.PhaseFunctionPtr)

        contribs = mi.Spectrum(0.0)
        mei_main = None
        for i in range(n_probes):
            mei_sub.t = dr.fma(alt_sampler.next_1d(active), sub_len, t0)
            mei_sub.p = dr.fma(seg_dir, mei_sub.t, seg_origin)

            with dr.suspend_grad():
                if i == 0:
                    # Snapshot of the main probe (the linear-cost variant may
                    # retain it in the reservoir for the deferred suffix).
                    mei_main = mi.MediumInteraction3f(mei_sub)
                    if include_indirect:
                        # Main probe: direct + one shared recursive suffix.
                        nee_Li, ind_Li = self._probe_radiance(
                            scene, medium, channel, alt_sampler, mei_sub,
                            phase_ctx, phase, nee_dir_sample, suffix_depth, within)
                        Li = nee_Li + n_probes * ind_Li
                    else:
                        # Linear-cost variant: direct light only; the indirect
                        # component is deferred to the reservoir-selected
                        # segment (see the caller).
                        Li = self._probe_direct(
                            scene, medium, channel, alt_sampler, mei_sub,
                            phase_ctx, phase, nee_dir_sample, within)
                else:
                    # Additional probes: direct lighting only (the indirect
                    # component is amortized over the segment by the factor
                    # `n_probes` above).
                    Li = self._probe_direct(
                        scene, medium, channel, alt_sampler, mei_sub,
                        phase_ctx, phase, nee_dir_sample, within)

            with dr.resume_grad():
                sigma_s_sub, _, sigma_t_sub = \
                    medium.get_scattering_coefficients(mei_sub, active)
                # Attached albedo derived from sigma_s / sigma_t: identical
                # value (where sigma_t = 0, the albedo factor below is
                # multiplied by detach(sigma_t) = 0 anyway) and saves a
                # separate get_albedo() vcall.
                albedo_sub = sigma_s_sub / dr.maximum(sigma_t_sub, 1e-8)

                if dr.hint(self.use_probe_mis, mode='scalar'):
                    # Complement of the vertex-side power-heuristic weight
                    mis_p = dr.rcp(1 + dr.square(dr.detach(sigma_t_sub)))
                else:
                    mis_p = mi.Spectrum(1.0)

                # Matched estimator: both terms below evaluate sigma_t at the
                # *same* location `mei_sub.p`.
                contribs -= sigma_t_sub * adj_trans
                contribs += (sigma_t_sub * dr.detach(albedo_sub)
                             + mis_p * dr.detach(sigma_t_sub) * albedo_sub) \
                            * adj_scatt * Li

        # Uniform probe placement: pdf = 1 / sub_len (per probe)
        inv_pdf = sub_len / n_probes
        with dr.resume_grad():
            if dr.hint(dr.grad_enabled(contribs), mode='scalar'):
                safe = active & dr.all(dr.isfinite(contribs))
                # This backward runs in an *evaluated* context (deferred
                # flush), unlike the in-loop backward calls above, which are
                # re-traced (and hence re-attached) on every render. The
                # default ADFlag.ClearEdges would delete the persistent
                # parameter->texture edges shared with subsequent renders in
                # the same session, silently zeroing their gradients from the
                # second render onward. Keep the graph; only clear vertex
                # gradients.
                dr.backward(dr.select(safe, contribs, 0.0) * inv_pdf,
                            flags=dr.ADFlag.ClearVertices)

        return mei_main

    def _probe_radiance(self, scene, medium, channel, alt_sampler, mei_sub,
                        phase_ctx, phase, nee_dir_sample, suffix_depth, active):
        """
        Estimate the in-scattered radiance at a probe location, decomposed
        into (direct NEE, indirect) components. The indirect component traces
        one detached suffix path by recursively invoking :py:meth:`sample` in
        primal mode.
        """
        nee_Li = self._probe_direct(scene, medium, channel, alt_sampler, mei_sub,
                                    phase_ctx, phase, nee_dir_sample, active)
        ind_Li = self._probe_indirect(scene, channel, alt_sampler, mei_sub,
                                      phase_ctx, phase, suffix_depth, active)
        return dr.select(active, nee_Li, 0.0), ind_Li

    def _probe_indirect(self, scene, channel, alt_sampler, mei_sub,
                        phase_ctx, phase, suffix_depth, active):
        """
        Indirect in-scattered radiance at a probe location: phase-sample a
        direction and trace one detached suffix path by recursively invoking
        :py:meth:`sample` in primal mode.
        """
        wo, phase_weight, phase_pdf = phase.sample(
            phase_ctx, mei_sub,
            alt_sampler.next_1d(active), alt_sampler.next_2d(active), active)
        rec_active = active & (phase_pdf > 0.0)
        rec_ray = mei_sub.spawn_ray(wo)

        last_scatter = dr.zeros(mi.Interaction3f)
        last_scatter[rec_active] = mei_sub
        state = _SuffixState(depth=mi.UInt32(suffix_depth),
                             medium=mi.MediumPtr(mei_sub.medium),
                             last_scatter_event=last_scatter,
                             last_scatter_direction_pdf=dr.select(rec_active, phase_pdf, 1.0),
                             channel=channel,
                             active=rec_active)
        Li, _, _, _ = self.sample(dr.ADMode.Primal, scene, alt_sampler, rec_ray,
                                  δL=None, state_in=None, active=rec_active,
                                  path_state=state)

        return dr.select(rec_active, phase_weight * Li, 0.0)

    def _probe_direct(self, scene, medium, channel, alt_sampler, mei_sub,
                      phase_ctx, phase, nee_dir_sample, active):
        """
        Direct lighting (NEE) at a probe location, using the segment's shared
        emitter direction sample and MIS against phase sampling.
        """
        emitted, ds = self.sample_emitter(
            mei_sub, dr.zeros(mi.SurfaceInteraction3f), active, mi.Bool(False),
            scene, alt_sampler, medium, channel, active,
            mode=dr.ADMode.Primal, dir_sample=nee_dir_sample)
        phase_val, phase_pdf = phase.eval_pdf(phase_ctx, mei_sub, ds.d, active)
        nee_directional_pdf = dr.select(ds.delta, 0.0, phase_pdf)
        return dr.select(active,
                         phase_val * mis_weight(ds.pdf, nee_directional_pdf) * emitted,
                         0.0)

    def to_string(self):
        return (f'PRBVolpathSMIntegrator[max_depth = {self.max_depth}, '
                f'probes_per_segment = {self.probes_per_segment}]')

mi.register_integrator("prbvolpath_sm", lambda props: PRBVolpathSMIntegrator(props))

del PRBVolpathIntegrator

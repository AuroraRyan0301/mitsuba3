"""
Tests for the sample-matching volumetric PRB integrator (`prbvolpath_sm`).

The integrator must (a) match `prbvolpath` exactly in primal mode (same
estimator), and (b) produce unbiased extinction gradients: its adjoint
directional derivatives are compared against those of `prbvolpath`, whose
gradients are validated independently in `test_ad_integrators.py`.
"""

import pytest
import drjit as dr
import mitsuba as mi


def make_scene(integrator_dict, grid, grid_scale=1.0):
    import numpy as np
    return mi.load_dict({
        'type': 'scene',
        'integrator': integrator_dict,
        'light': {'type': 'constant', 'radiance': 1.0},
        'sensor': {
            'type': 'perspective', 'fov': 45,
            'to_world': mi.ScalarTransform4f().look_at([0, 0, 4], [0, 0, 0], [0, 1, 0]),
            'film': {'type': 'hdrfilm', 'width': 16, 'height': 16,
                     'rfilter': {'type': 'box'}, 'pixel_format': 'rgb'},
            'sampler': {'type': 'independent', 'sample_count': 8},
        },
        'medium_box': {
            'type': 'cube', 'bsdf': {'type': 'null'},
            'interior': {
                'type': 'heterogeneous',
                'sigma_t': {
                    'type': 'gridvolume',
                    'grid': mi.VolumeGrid((grid * grid_scale).astype(np.float32)),
                    'to_world': mi.ScalarTransform4f().translate([-1, -1, -1]).scale(2.0),
                },
                'albedo': 0.8, 'scale': 2.0,
            },
        },
        # An opaque surface inside the medium exercises the mixed
        # surface/medium code path and the probe segment bookkeeping.
        'sphere': {
            'type': 'sphere', 'radius': 0.35,
            'to_world': mi.ScalarTransform4f().translate([0.5, 0, 0]),
            'bsdf': {'type': 'diffuse',
                     'reflectance': {'type': 'rgb', 'value': [0.5, 0.5, 0.5]}},
        },
    })


def make_grid():
    import numpy as np
    rng = np.random.default_rng(0)
    return rng.uniform(0.4, 1.6, size=(4, 4, 4, 1)).astype(np.float32)


def adjoint_dirderiv(scene, grid, n_seeds, spp):
    """Directional derivative of mean(image) along the sigma_t grid values."""
    import numpy as np
    params = mi.traverse(scene)
    key = [k for k in params.keys() if 'sigma_t' in k and k.endswith('.data')][0]
    vals = []
    for s in range(n_seeds):
        dr.enable_grad(params[key])
        params.update()
        img = mi.render(scene, params, spp=spp, seed=10 + s, seed_grad=1000 + s)
        dr.backward(dr.mean(img, axis=None))
        g = np.array(dr.grad(params[key])).ravel()
        dr.disable_grad(params[key])
        assert np.isfinite(g).all(), 'non-finite gradients'
        vals.append(float((g * grid.ravel()).sum()))
    return np.array(vals)


def test01_primal_matches_prbvolpath(variants_all_ad_rgb_unpolarized):
    import numpy as np
    grid = make_grid()
    img_ref = np.array(mi.render(
        make_scene({'type': 'prbvolpath', 'max_depth': 4}, grid), spp=32, seed=3))
    img_sm = np.array(mi.render(
        make_scene({'type': 'prbvolpath_sm', 'max_depth': 4}, grid), spp=32, seed=3))
    assert np.isfinite(img_sm).all()
    assert np.allclose(img_sm.mean(), img_ref.mean(), rtol=0.05)


@pytest.mark.parametrize('config', [
    {},                                            # quadratic, one probe
    {'probes_per_segment': 4},                     # quadratic, multi-probe
    {'linear_cost': True},                         # linear (reservoir)
])
def test02_gradients_match_prbvolpath(variants_all_ad_rgb_unpolarized, config):
    import numpy as np
    grid = make_grid()
    n_seeds, spp = 4, 32

    v_ref = adjoint_dirderiv(
        make_scene({'type': 'prbvolpath', 'max_depth': 4}, grid), grid, n_seeds, spp)
    v_sm = adjoint_dirderiv(
        make_scene({'type': 'prbvolpath_sm', 'max_depth': 4, **config}, grid),
        grid, n_seeds, spp)

    # Means must agree within Monte Carlo noise (both estimate the same
    # derivative; a bug of the kind sample matching could introduce shows up
    # as a large multiplicative bias, far outside this tolerance).
    m_ref, m_sm = v_ref.mean(), v_sm.mean()
    sigma = (v_ref.std() + v_sm.std()) / (n_seeds ** 0.5)
    assert abs(m_sm - m_ref) < max(4 * sigma, 0.15 * abs(m_ref)), \
        f'gradient mismatch: sm={m_sm:.4e} ref={m_ref:.4e} (config={config})'


def test03_medium_get_albedo(variants_all_ad_rgb_unpolarized):
    """get_albedo must agree with sigma_s / sigma_t for the built-in media."""
    for medium_dict in (
        {'type': 'homogeneous', 'sigma_t': 2.0, 'albedo': 0.8},
        {'type': 'heterogeneous', 'sigma_t': 1.5, 'albedo': 0.3, 'scale': 2.0},
    ):
        medium = mi.load_dict(medium_dict)
        mei = dr.zeros(mi.MediumInteraction3f)
        mei.p = mi.Point3f(0.5, 0.5, 0.5)
        sigma_s, _, sigma_t = medium.get_scattering_coefficients(mei)
        albedo = medium.get_albedo(mei)
        assert dr.allclose(albedo, sigma_s / dr.maximum(sigma_t, 1e-8))
    # The vectorized MediumPtr variant must also expose it
    assert hasattr(mi.MediumPtr, 'get_albedo')

"""
Tests for the sample-matching volumetric integrator `prbvolpath_sm`.

In primal mode it must match `prbvolpath` exactly. Its extinction gradients
are compared against those of `prbvolpath`, which `test_ad_integrators.py`
validates independently.
"""

import gc

import pytest
import drjit as dr
import mitsuba as mi


SM_CONFIGS = [
    {'type': 'prbvolpath_sm'},
    {'type': 'prbvolpath_sm', 'gradient_samples_per_segment': 4},
    {'type': 'prbvolpath_sm_linear'},
]


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


def adjoint_dirderiv(scene, grid, n_seeds, spp, direction=None):
    """Directional derivative of mean(image) along the sigma_t grid values.

    `direction` defaults to the grid itself; pass an explicit one when the
    stored parameter is not the extinction (see `scale_direction`).
    """
    import numpy as np
    params = mi.traverse(scene)
    key = [k for k in params.keys() if 'sigma_t' in k and k.endswith('.data')][0]
    if direction is None:
        direction = grid
    direction = np.asarray(direction).ravel()
    vals = []
    for s in range(n_seeds):
        dr.enable_grad(params[key])
        params.update()
        img = mi.render(scene, params, spp=spp, seed=10 + s, seed_grad=1000 + s)
        dr.backward(dr.mean(img, axis=None))
        g = np.array(dr.grad(params[key])).ravel()
        dr.disable_grad(params[key])
        assert np.isfinite(g).all(), 'non-finite gradients'
        vals.append(float((g * direction).sum()))
    return np.array(vals)


def test01_primal_matches_prbvolpath(variants_all_ad_rgb_unpolarized):
    import numpy as np
    grid = make_grid()
    img_ref = np.array(mi.render(
        make_scene({'type': 'prbvolpath', 'max_depth': 4}, grid), spp=32, seed=3))
    img_sm = np.array(mi.render(
        make_scene({'type': 'prbvolpath_sm', 'max_depth': 4}, grid),
        spp=32, seed=3))
    assert np.isfinite(img_sm).all()
    assert np.allclose(img_sm.mean(), img_ref.mean(), rtol=0.05)


@pytest.mark.parametrize('config', SM_CONFIGS)
def test02_gradients_match_prbvolpath(variants_all_ad_rgb_unpolarized, config):
    import numpy as np
    grid = make_grid()
    n_seeds, spp = 4, 32

    v_ref = adjoint_dirderiv(
        make_scene({'type': 'prbvolpath', 'max_depth': 4}, grid), grid, n_seeds, spp)
    v_sm = adjoint_dirderiv(
        make_scene({'max_depth': 4, **config}, grid),
        grid, n_seeds, spp)

    # Both estimate the same derivative, so the means must agree within Monte
    # Carlo noise. A bias in the estimator shows up as a large factor.
    m_ref, m_sm = v_ref.mean(), v_sm.mean()
    sigma = (v_ref.std() + v_sm.std()) / (n_seeds ** 0.5)
    assert abs(m_sm - m_ref) < max(4 * sigma, 0.15 * abs(m_ref)), \
        f'gradient mismatch: sm={m_sm:.4e} ref={m_ref:.4e} (config={config})'


def make_grid_rgb():
    """Three-channel extinction; one channel would see the same value at every wavelength"""
    import numpy as np
    rng = np.random.default_rng(0)
    g = rng.uniform(0.4, 1.6, size=(4, 4, 4, 3)).astype(np.float32)
    g[..., 1] *= 0.6
    g[..., 2] *= 1.4
    return g


def scale_direction(scene):
    """Perturbation direction. Spectral variants store three channels as sRGB
    coefficients (c0, c1, c2, scale); only `scale` gives a measurable derivative."""
    import numpy as np
    params = mi.traverse(scene)
    key = [k for k in params.keys() if 'sigma_t' in k and k.endswith('.data')][0]
    d = np.zeros(params[key].shape, dtype=np.float32)
    if mi.is_spectral:
        d[..., -1] = 1.0   # the sRGB model's scale
    else:
        d[...] = 1.0       # plain RGB channels
    return d


def test03_primal_matches_prbvolpath_spectral(variants_all_ad_spectral):
    """A three-channel extinction must render the same in a spectral variant,
    where the deferred records also have to carry the path's wavelengths."""
    if mi.is_polarized:
        pytest.skip('Neither this integrator nor `prbvolpath`, the reference '
                    'it is compared against, supports polarized rendering.')
    import numpy as np
    grid = make_grid_rgb()
    img_ref = np.array(mi.render(
        make_scene({'type': 'prbvolpath', 'max_depth': 4}, grid), spp=32, seed=3))
    img_sm = np.array(mi.render(
        make_scene({'type': 'prbvolpath_sm', 'max_depth': 4}, grid),
        spp=32, seed=3))
    assert np.isfinite(img_sm).all()
    assert np.allclose(img_sm.mean(), img_ref.mean(), rtol=0.05)


@pytest.mark.parametrize('config', SM_CONFIGS)
def test04_gradients_match_prbvolpath_spectral(variants_all_ad_spectral, config):
    """Extinction gradients in a spectral variant. Without the wavelengths in
    the deferred record the probe kernel evaluates the medium at wavelength
    zero, which does not raise -- it silently biases the gradient."""
    if mi.is_polarized:
        pytest.skip('Neither this integrator nor `prbvolpath`, the reference '
                    'it is compared against, supports polarized rendering.')
    import numpy as np
    grid = make_grid_rgb()
    n_seeds, spp = 4, 64

    scene_ref = make_scene({'type': 'prbvolpath', 'max_depth': 4}, grid)
    direction = scale_direction(scene_ref)

    v_ref = adjoint_dirderiv(scene_ref, grid, n_seeds, spp, direction)

    # Release the reference scene so that only one Medium instance is alive
    del scene_ref
    gc.collect()
    v_sm = adjoint_dirderiv(
        make_scene({'max_depth': 4, **config}, grid),
        grid, n_seeds, spp, direction)

    m_ref, m_sm = v_ref.mean(), v_sm.mean()
    sigma = (v_ref.std() + v_sm.std()) / (n_seeds ** 0.5)
    assert abs(m_sm - m_ref) < max(4 * sigma, 0.15 * abs(m_ref)), \
        f'gradient mismatch: sm={m_sm:.4e} ref={m_ref:.4e} (config={config})'


@pytest.mark.parametrize('config', SM_CONFIGS)
def test05_gradients_colored_extinction(variants_all_ad_rgb_unpolarized, config):
    """Extinction that varies across channels. All channels share one acceptance
    test, so the gradient needs a per-channel factor that a gray grid never exposes."""
    import numpy as np
    grid = make_grid_rgb()
    n_seeds, spp = 4, 64

    scene_ref = make_scene({'type': 'prbvolpath', 'max_depth': 4}, grid)
    direction = scale_direction(scene_ref)

    v_ref = adjoint_dirderiv(scene_ref, grid, n_seeds, spp, direction)

    # Release the reference scene so that only one Medium instance is alive
    del scene_ref
    gc.collect()
    v_sm = adjoint_dirderiv(make_scene({'max_depth': 4, **config}, grid),
                            grid, n_seeds, spp, direction)

    m_ref, m_sm = v_ref.mean(), v_sm.mean()
    sigma = (v_ref.std() + v_sm.std()) / (n_seeds ** 0.5)
    assert abs(m_sm - m_ref) < max(4 * sigma, 0.15 * abs(m_ref)), \
        f'gradient mismatch: sm={m_sm:.4e} ref={m_ref:.4e} (config={config})'

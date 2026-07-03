"""
Tests for the majorant supergrid (`majorant_resolution_factor`).

The supergrid only changes how free-flight distances are sampled (a
spatially-varying majorant with DDA traversal instead of a single global
majorant); it must not change what is being estimated. Primal renders and
adjoint gradients must therefore agree with the global-majorant baseline
within Monte Carlo noise.
"""

import pytest
import drjit as dr
import mitsuba as mi


def make_grid():
    import numpy as np
    rng = np.random.default_rng(0)
    grid = rng.uniform(0.2, 0.8, size=(8, 8, 8, 1)).astype(np.float32)
    # An outlier voxel: with a global majorant this taxes every ray in the
    # volume; the supergrid must localize it (and stay unbiased).
    grid[1, 1, 1, 0] = 4.0
    return grid


def make_scene(grid, factor):
    import numpy as np
    medium = {
        'type': 'heterogeneous',
        'sigma_t': {
            'type': 'gridvolume',
            'grid': mi.VolumeGrid(grid),
            'to_world': mi.ScalarTransform4f().translate([-1, -1, -1]).scale(2.0),
        },
        'albedo': 0.8, 'scale': 2.0,
    }
    if factor:
        medium['majorant_resolution_factor'] = factor
        medium['majorant_factor'] = 1.01
    return mi.load_dict({
        'type': 'scene',
        'integrator': {'type': 'volpath', 'max_depth': 16},
        'light': {'type': 'constant', 'radiance': 1.0},
        'sensor': {
            'type': 'perspective', 'fov': 45,
            'to_world': mi.ScalarTransform4f().look_at([0, 0, 4], [0, 0, 0], [0, 1, 0]),
            'film': {'type': 'hdrfilm', 'width': 16, 'height': 16,
                     'rfilter': {'type': 'box'}, 'pixel_format': 'rgb'},
            'sampler': {'type': 'independent', 'sample_count': 16},
        },
        'medium_box': {'type': 'cube', 'bsdf': {'type': 'null'}, 'interior': medium},
    })


@pytest.mark.parametrize('factor', [1, 2, 4])
def test01_primal_parity(variants_vec_backends_once_rgb, factor):
    import numpy as np
    grid = make_grid()
    means = {}
    for f in (0, factor):
        imgs = [np.array(mi.render(make_scene(grid, f), spp=64, seed=s))
                for s in range(4)]
        img = np.mean(imgs, axis=0)
        assert np.isfinite(img).all()
        means[f] = img.mean()
    assert np.allclose(means[0], means[factor], rtol=0.03), \
        f'supergrid factor={factor} changed the primal render: {means}'


def test02_gradient_parity(variants_all_ad_rgb_unpolarized):
    import numpy as np
    grid = make_grid()

    def dirderiv(factor, n=4):
        scene = make_scene(grid, factor)
        # volpath is not differentiable; use the AD integrator
        integrator = mi.load_dict({'type': 'prbvolpath', 'max_depth': 16})
        params = mi.traverse(scene)
        key = [k for k in params.keys() if k.endswith('sigma_t.data')][0]
        vals = []
        for s in range(n):
            dr.enable_grad(params[key]); params.update()
            img = mi.render(scene, params, integrator=integrator,
                            spp=32, seed=10 + s, seed_grad=1000 + s)
            dr.backward(dr.mean(img, axis=None))
            g = np.array(dr.grad(params[key])).ravel()
            dr.disable_grad(params[key])
            assert np.isfinite(g).all()
            vals.append(float((g * grid.ravel()).sum()))
        return np.array(vals)

    v0, v2 = dirderiv(0), dirderiv(2)
    sigma = (v0.std() + v2.std()) / 2.0
    assert abs(v0.mean() - v2.mean()) < max(4 * sigma, 0.15 * abs(v0.mean())), \
        f'gradients diverge: off={v0.mean():.4e} on={v2.mean():.4e}'


def test03_homogeneous_rejects_supergrid(variants_all_rgb_unpolarized):
    with pytest.raises(RuntimeError, match='majorant'):
        mi.load_dict({'type': 'homogeneous', 'sigma_t': 1.0,
                      'majorant_resolution_factor': 4})

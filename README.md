Figures for the sample-matching branch (`sample-matching-rework-rebased`).

fig_grad_mean, fig_grad_var: per-voxel extinction-gradient mean and variance at
equal iteration count, the DRT reference against sample matching at three
record-slot counts. Bunny cloud, 768x576, uniform white environment, 8 spp,
N = 512 adjoint renders with identical seeds, gradient_samples_per_segment 4.
Rows carry the cost in path length. The DRT reference runs with subsampling off,
its quadratic setting, so it is compared against a record slot per segment like
for like. Variance rather than standard deviation, to match the paper.

    estimator                              <g,1>       var occupied   var all
    DRT reference, quadratic          -1.18298e8          136.5        318.9
    one record slot per lane          -1.18282e8           61.8         91.5
    four record slots                 -1.18284e8           46.4         81.4
    a slot per segment, quadratic     -1.18284e8           45.4         80.9

fig_training: test PSNR of the bunny reconstruction against iterations and wall
time, one A40 each. 63 training views, 768x576, batches of 32768 pixels, spp
1024 / 16, L1, Adam 6e-3, 64^3 -> 128^3 -> 256^3, max depth 64, no Russian
roulette, 8000 iterations, majorant_factor 1.2. Circles mark checkpoint resumes,
which reset Adam's moments.

    integrator              test PSNR     wall time
    prbvolpath               33.8 dB        40.9 h
    prbvolpath_sm            38.2 dB        25.7 h
    prbvolpath_sm_linear     35.7 dB        22.6 h

The PDFs carry vector text.

# Figure 6(a) — independently computed

Command: `python tools/reproduce_figure6a.py --compute`

MAE vs derived experiment (30/40/60/180 µm) = 0.05215 cycles/mm
RMSE = 0.05465 cycles/mm

Simulation sources: aligned_reference_convention, analytical_phase_gradient, image_center_spacing_20_percent, image_center_spacing_20_percent, image_center_spacing_20_percent, image_center_spacing_20_percent

Figure 6(a). Simulated and experimental transverse interference patterns obtained at idler-lens displacements of 0, 20, 30, 40, 60, and 180 µm. The corresponding spatial-frequency curves are based on the mean separation between adjacent resolved bright-fringe centers. For the simulated images, a uniform relative prominence threshold of 20% was applied. At 20 µm, the simulated intensity distribution contains fewer than two resolvable bright-fringe centers; therefore, the plotted simulated frequency is obtained from the analytical phase gradient of the corrected transverse optical model rather than direct image-based fringe spacing. The zero-displacement point represents the aligned-reference convention. All remaining nonzero frequencies are extracted directly from simulated or experimental intensity distributions.

Simulation values are independently computed. Experimental frequency markers are derived Table S3 public records. Reference Final_Data.csv was not used as a computational substitute.

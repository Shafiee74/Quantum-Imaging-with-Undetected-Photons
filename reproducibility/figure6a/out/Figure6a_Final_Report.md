# Figure 6(a) — public reproduction

Command: `python tools/reproduce_figure6a.py --public`

MAE = 0.05215 cycles/mm; RMSE = 0.05465 cycles/mm (image-derived nonzero points; excludes analytical 20 µm).
Including analytical 20 µm: MAE = 0.06186; RMSE = 0.06647.
Analytical 20 µm simulation point uses a hollow circular marker.

Figure 6(a). Simulated and experimental transverse interference patterns obtained at idler-lens displacements of 0, 20, 30, 40, 60, and 180 µm. The corresponding spatial-frequency curves are based on the mean separation between adjacent resolved bright-fringe centers. For the simulated images, a uniform relative prominence threshold of 20% was applied. At 20 µm, the simulated intensity distribution contains fewer than two resolvable bright-fringe centers; therefore, the plotted simulated frequency is obtained from the analytical phase gradient of the corrected transverse optical model rather than direct image-based fringe spacing. The zero-displacement point represents the aligned-reference convention. All remaining nonzero frequencies are extracted directly from simulated or experimental intensity distributions.

Experimental image strip omitted: raw EMCCD TIFFs are restricted. Frequency markers use reproducibility/figure6a derived CSV. Hollow circle = analytical 20 µm simulation point.

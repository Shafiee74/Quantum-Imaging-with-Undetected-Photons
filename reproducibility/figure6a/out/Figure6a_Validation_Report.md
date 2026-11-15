# Figure 6(a) validation — **PASS**

Computed: `D:\Quantum\Quantum\QIUP_Synthetic_Dataset\Ver1.3.4(Digital_Twin_Verification)\DT_version 3.4.2_12Nov\reproducibility\figure6a\computed\Figure6a_Computed_Data.csv`
Reference (read-only): `D:\Quantum\Quantum\QIUP_Synthetic_Dataset\Ver1.3.4(Digital_Twin_Verification)\DT_version 3.4.2_12Nov\reproducibility\figure6a\reference\Figure6a_Final_Data.csv`

MAE (30/40/60/180) = 0.05215 (expected ≈ 0.05215)
RMSE = 0.05465 (expected ≈ 0.05465)
Max |Δk_sim| vs reference = 4.340e-10

| Δx | computed k | reference k | abs err | centers c/r | pass |
|---:|---:|---:|---:|---|---|
| 0 | 0.000000000 | 0.000000000 | 0.000e+00 | 1/1 | PASS |
| 20 | 0.329218107 | 0.329218107 | 4.115e-12 | 1/1 | PASS |
| 30 | 0.584966364 | 0.584966364 | 4.340e-10 | 3/3 | PASS |
| 40 | 0.722282412 | 0.722282412 | 4.233e-10 | 3/3 | PASS |
| 60 | 1.005530417 | 1.005530417 | 2.951e-10 | 3/3 | PASS |
| 180 | 2.970002970 | 2.970002970 | 2.970e-12 | 11/11 | PASS |

All predetermined tolerances satisfied. Simulation independently reproduced.

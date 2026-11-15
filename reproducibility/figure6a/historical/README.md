# Historical / obsolete Figure 6(a) assets

Earlier simulation tables (e.g. frequencies ≈ 0.70, 0.97, 1.29, 1.92, 5.68
cycles/mm and simulated center counts 5/7/21 at 40/60/180 µm) belong to a
**pre-correction** extraction gate and must not drive the active pipeline.

The authoritative manuscript set uses simulated centers 1/1/3/3/3/11 and the
final frequencies documented in `../reference/Figure6a_Final_Data.csv`.

## Quarantined tooling (not on the publication path)

| Script | Role after quarantine |
|---|---|
| `tools/figure6a_simulated_fringe_frequency_correction.py` | Library (`bright_crests`) for the public pipeline; `__main__` blocked unless `--allow-historical` (writes only under this folder) |
| `tools/figure6a_manuscript_candidate.py` | Historical candidate helper; `__main__` blocked unless `--allow-historical` |

Do **not** cite `Figure6a_Corrected*` outputs as the submitted figure.
Do **not** link README examples or verification scripts to historical tables.

## Authoritative public commands

```bash
python tools/reproduce_figure6a.py --compute
python tools/reproduce_figure6a.py --verify
```

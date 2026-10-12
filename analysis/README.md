# analysis/

Outputs used in the paper.

| Path | Contents |
|---|---|
| `T_*.csv` | Seed-level summaries, confidence intervals and Holm-corrected tests (explainer and condition contrasts, stability, runtime, partition sensitivity) |
| `power.json` | Minimum detectable effect for the paired tests |
| `seed_level_explanations.csv` | Seed-level faithfulness and stability outcomes used for inference |
| `extras_*.csv` | Copies of the secondary analyses |
| `audit_all.json` | Combined dataset audit |
| `figures/` | Figures 1–7 of the paper (`fig_*.pdf`, PNG previews) |
| `tables/` | LaTeX bodies of Tables 4–12 and `numbers.json` (every number quoted in the text) |

Regenerate:

```bash
python -m iotxai.analysis ./results --out ./analysis
python tools/make_figures.py ./results ./analysis ./analysis/figures
python tools/make_tables.py  ./results ./analysis ./analysis/tables
```

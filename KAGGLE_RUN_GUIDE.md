# How to run the revision experiments on Kaggle

You need **one Kaggle notebook** (`notebooks/kaggle_revision_v2.ipynb`). You will save **five versions**
of it, one per session. Each version runs in the background, so you can close the browser.

## 1. One-time setup (5 minutes)

1. Go to kaggle.com → **Create → New Notebook → File → Import Notebook** and upload `kaggle_revision_v2.ipynb`.
2. Right panel → **Add Input**: add the same two datasets you used before:
   * ToN-IoT network dataset (the folder that contains `train_test_network.csv`)
   * Edge-IIoTset (`mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot`)
3. Right panel → **Session options**: Accelerator = **GPU T4 x1**, Internet = **On**, Persistence = Files.

## 2. Session TEST: check that everything works (about 15 minutes, run interactively)

* Find the cell that starts with `# ===== SESSION =====` and leave `SESSION = "TEST"`.
* Click **Run All**.
* At the end you should see two tables (predictive metrics and attack metrics) and `results_session_TEST.zip`.
* **Send me the TEST zip (or a screenshot of the two tables) before starting A/B.** Then I can check the
  Edge-IIoTset baseline and the feature audit on the real data before you spend GPU hours.

If a cell fails, send me the full error text. If the datasets are not found automatically, paste the
full CSV paths into `TON_CSV = "..."` and `EDGE_CSV = "..."`.

## 3. Full sessions: one saved version each

For each of `A`, `B`, `C1`, `C2`:

1. Set `SESSION = "A"` (then `"B"`, `"C1"`, `"C2"`).
2. Top right **Save Version → Save & Run All (Commit) → Save**.
3. Change `SESSION` and save the next version right away. Kaggle queues versions or runs them in parallel,
   depending on your quota.

| Session | What it runs | Estimated T4 time |
|---|---|---|
| A  | ToN-IoT, 20 seeds, 4 detectors, all attacks, SHAP/LIME, repeated-call checks | ~3–5 h |
| B  | Edge-IIoTset, same as A | ~3–5 h |
| C1 | 2 extra data partitions × 5 seeds, both datasets | ~2–3 h |
| C2 | Adversarial-training trade-off sweep, baseline ablation (feature policy × de-duplication × threshold), multiclass analysis, memory/latency profiling | ~4–5 h |
| C3 | Attack success at false-positive-rate-matched thresholds (5 seeds, both datasets) | ~1 h |

The total is about 12–17 GPU hours, which fits in the free 30 h/week quota. A version may stop at Kaggle's
12-hour limit. If that happens, create a new version: add the stopped version's output as an input and set
`RESUME_FROM = "/kaggle/input/<that-output>/results_v2"`. Finished runs are skipped.

## 4. Return the results

When a version finishes, open it → **Output** → download `results_session_<X>.zip`. Send all five result zips (A, B, C1, C2, C3)
(A, B, C1, C2, C3) to the person running the analysis, who will update the tables and figures and finish
the response letter.

## 5. Public repository (do this in parallel)

1. Create a public GitHub repository, e.g. `iotxai-revision`, and upload the contents of the `iotxai_repo`
   folder I sent (the result folders are added after the runs finish).
2. Connect GitHub to Zenodo (zenodo.org → GitHub → enable the repository), then create a **GitHub Release**
   `v2.0.0`. Zenodo then issues a DOI automatically.
3. Send me the GitHub URL and the DOI so I can put them in the Data/Code Availability statement.

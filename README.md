# Reproducible disagreement recovery

Code and derived results for the article
"How Much Reproducible Human Disagreement Can Text Models Recover in Emotion, Inference, and Safety Annotation?"

The analysis measures how well item-level annotator disagreement repeats across disjoint annotator groups,
maps model predictions to the expected disagreement of a group of the same size, and normalizes the
model-human agreement by the human reproducibility (the recoverable share `R_text = m_E^2 / r_nn`).

## Contents

- `code/` analysis and training scripts. They were run with Python 3.10.12, PyTorch 2.7.1, Transformers 4.53.2, scikit-learn 1.7.2, NumPy 2.2.6, and pandas 2.3.3.
- `results/` JSON outputs from which every number in the article is taken.
- `data/split/` train, development, and test identifiers: `splits.json` (BRIGHTER English), `splits_e2.json` (ChaosNLI, DICES-990), and `splits_j.json` (BRIGHTER German, DICES-350). The scripts regenerate them with fixed seeds.

Raw datasets are not redistributed. Obtain them from their providers and place them as follows:

| Dataset | Expected file(s) under `data/raw/` |
|---|---|
| BRIGHTER English, per-annotator release | `eng.csv` |
| BRIGHTER German, per-annotator release | `deu_individuals.csv` |
| ChaosNLI | `chaosnli/raw/chaosNLI_snli.jsonl`, `chaosnli/raw/chaosNLI_mnli_m.jsonl` |
| DICES-350 and DICES-990 | `dices_350.csv`, `dices_990.csv` |

## Pipeline

Run from `code/`. GPU steps train RoBERTa-large, its MNLI checkpoint, XLM-RoBERTa-large, or extract
frozen Qwen2.5-7B-Instruct embeddings; CPU steps only read saved predictions.

| Step | Command | Output used for |
|---|---|---|
| Votes | `python s1_votes.py --raw ../data/raw --out ../data/votes` | all |
| Reproducibility | `python s2_reliability.py --votes ../data/votes --out ../results/s2_reliability.json` | Spearman-Brown check |
| BRIGHTER English models | `bash run_s3.sh` | Table 2 (full) |
| ChaosNLI, DICES-990, vote thinning | `bash run_s4.sh` | Table 2, Section 5.5 |
| Matched votes, no MNLI checkpoint | `bash run_s5.sh` | Tables 2, 4, 6 |
| Matched training size | `bash run_s6.sh` | Tables 2, 4 |
| Direct regression | `bash run_s9.sh` | Table 4 |
| Comparison predictors, second datasets | `bash run_s13.sh` | Tables 3, 4 |
| Mapping and `R_text` | `s3_nstar.py`, `s4_map.py`, `s5_map.py`, `s6_map.py`, `s9_map.py` | Tables 2, 4, 6 |
| Partition | `s7_decomp.py` | Figure 2 |
| One-hot votes | `s8_onehot.py` | Section 5.6, Table 6 |
| Intervals, group sizes, redraws | `s10_explore.py` | Tables 2, 3, 5, Section 5.6 |
| German pool check | `s16_deu_check.py` | Section 5.3 |
| Figures | `s17_fig1_overview.py`, `s18_fig_partition_v2.py` | Figures 1, 2 |

The usage of each script is given in its docstring. Keys such as `H1` to `H13` and `J1` to `J5` in the result files are labels
of individual analyses; the article reports the underlying values, not these labels.

## Random seeds

Splits use seed 0. Model training uses seeds 0, 1, and 2, and the averaged prediction is evaluated.
Vote thinning, one-hot redraws, finite-group targets, and bootstraps use the fixed seeds stated in each script.

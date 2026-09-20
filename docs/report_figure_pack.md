# Canonical report figure pack

This branch adds `scripts/14_make_report_figures.py`, which renders the seven figures selected for the final written report from the already-frozen result CSVs. It does **not** rerun feature extraction, SFS, classifiers, or validation.

## Why these seven figures

1. **Baseline intra vs inter** — establishes the subject-generalization gap.
2. **Baseline decomposition + asymmetry extension** — separates BP/COH decomposition from the asymmetry hypothesis.
3. **Initial PLV gain + held-out-subject consistency** — shows that the first PLV improvement is broad across held-out participants.
4. **PLV decomposition including all feature families** — identifies the source of the PLV gain and explicitly shows the 312-candidate all-family condition.
5. **Six-band PLV validation + initial alpha recurrence** — shows why alpha was tested and that all six bands were evaluated.
6. **Zero-lag sensitivity** — compares baseline, all-band PLV, alpha PLV, and iPLV.
7. **Behavioral context** — shows Type1/Type2 response-time and attempts-to-30-correct differences.

All classification panels keep the three classifiers separate. Error bars are the existing standard deviations across participants / held-out participants from the formal result tables. A dashed horizontal line marks binary chance level (0.5).

For feature-set figures, labels such as `BP+COH+ASYM+PLV` mean the **candidate feature pool**. Every formal condition still uses the same training-only SFS before classification; the script writes this into the relevant figure.

## Run

From the repository root:

```bash
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/14_make_report_figures.py
```

Canonical report output:

```text
results/report_figures/report/
├── fig01_baseline_intra_vs_inter.png/.svg
├── fig02_baseline_decomposition_asymmetry.png/.svg
├── fig03_plv_initial_gain_subject_consistency.png/.svg
├── fig04_plv_decomposition_all_feature_families.png/.svg
├── fig05_plv_band_validation_alpha_recurrence.png/.svg
├── fig06_zero_lag_sensitivity.png/.svg
├── fig07_behavioral_context.png/.svg
└── figure_manifest.csv
```

SVG is recommended for the Word/PDF report because it remains vector-based. PNG is also produced for quick inspection.

## Optional layout variants

If the canonical report sizing looks cramped on a specific machine, the script can generate two alternate parameter sets without changing any data:

```bash
python scripts/14_make_report_figures.py --variant all
```

This additionally creates:

```text
results/report_figures/slide/
results/report_figures/compact/
```

Use `report/` for the canonical submission unless an alternate layout is visibly better. The three variants change only figure dimensions, font sizes, and DPI; plotted values and statistical summaries are identical.

## Data sources

The script reads only committed formal outputs:

- `results/intra/intra_overall_summary.csv`
- `results/inter/inter_summary.csv`
- `results/feature_set_comparison.csv`
- `results/extensions/extension_summary.csv`
- `results/extensions/extension_delta_by_unit.csv`
- `results/extensions/e2_plv/inter/inter_feature_frequency.csv`
- `results/plv_decomposition/plv_decomposition_summary.csv`
- `results/final_validation/final_validation_summary.csv`
- `results/behavioral_summary.csv`

No raw EEG is required.

## Figure-specific interpretation guardrails

- Figure 2 is **not** a full factorial comparison of BP, COH, and asymmetry. It shows baseline decomposition followed by the asymmetry extension.
- Figure 4 includes the all-family condition `BP+COH+ASYM+PLV` (312 candidate features), but all conditions still pass through SFS. It is not a no-SFS/all-columns experiment.
- Figure 5 uses the **initial +PLV experiment's** SFS recurrence to motivate alpha, then shows the later six-band validation. This preserves the actual analysis chronology.
- Figure 5's network is a sensor-pair feature-selection schematic; edges do not imply causal or anatomical connectivity.
- Figure 6 is titled zero-lag **sensitivity** because the iPLV result does not preserve the PLV gain.
- Figure 7 is behavioral context only; task difficulty was not independently randomized or rated, so it should not be used as a causal explanation for the EEG result.

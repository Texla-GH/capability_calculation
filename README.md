# Texla — Capability Calculation

Python console program for step-by-step process capability analysis using an Excel export from Infor M3.

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

Place `capability_test_20261103.xlsx` inside the `data/` directory (available in the draft ZIP shared in ChatGPT). The workbook is **not yet committed to GitHub**. Select the numbered menu stages in order.

## Analysis stages

1. Import Excel workbook (`Export` sheet).
2. Select item and test.
3. Validate 25 manufacturing orders, 125 individual measurements and 25 average records.
4. Recalculate means and ranges from each five-reading subgroup and compare them to the recorded M3 averages.
5. Histogram, Q-Q plot and Anderson–Darling normality check.
6. X-bar and R charts with statistical control limits.
7. Calculate Cp/Cpk with within-subgroup sigma and Pp/Ppk with overall sample sigma.

Chart images are written to `outputs/`.

## Validation notes

This is a first draft for staged validation. Control chart checks currently detect out-of-control points beyond limits, but do not yet implement supplementary run/trend rules. The workbook's recorded averages differ from averages recalculated from individual values; numerical conclusions must be reviewed before production use. A high Cpk alone does not establish capability if stability or distribution assumptions are not supported.

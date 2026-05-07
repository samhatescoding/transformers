# Comparison

This folder compares model results saved in `results/` and generates a larger report than the older one-off `results/analysis` charts.

It produces:

- benchmark-level score tables
- family-level summaries
- pairwise win/loss comparisons
- bootstrap confidence intervals across benchmarks
- ranking tables
- multiple plots in `comparison/output/`

Run:

```powershell
python comparison\build_report.py
```

Optional flags:

```powershell
python comparison\build_report.py --results-dir results --output-dir comparison\output --bootstrap-samples 5000
```

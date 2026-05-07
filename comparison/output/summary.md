# Comparison Summary

- Best overall mean benchmark-normalized score: `qwen25-vl` (0.865)
- Best raw mean score: `qwen25-vl`
- Most first-place finishes: `qwen25-vl`
- Benchmarks with estimated telemetry: `0` of `63`

## Models

- `qwen25-vl`: mean_score=0.519, normalized=0.865, mean_rank=1.24, wins=16, estimated_stats_benchmarks=0
- `small-llava`: mean_score=0.279, normalized=0.371, mean_rank=2.10, wins=6, estimated_stats_benchmarks=0
- `gemma`: mean_score=0.286, normalized=0.366, mean_rank=2.24, wins=4, estimated_stats_benchmarks=0

## Strongest Family Leaders

- `captioning`: `qwen25-vl` with mean_score=0.338
- `qa`: `gemma` with mean_score=0.600
- `labeling`: `qwen25-vl` with mean_score=0.645
- `detection`: `qwen25-vl` with mean_score=0.116

## Pairwise Edges

- `qwen25-vl` vs `gemma`: win_rate=0.833, mean_delta=0.233 across 21 benchmarks
- `qwen25-vl` vs `small-llava`: win_rate=0.786, mean_delta=0.240 across 21 benchmarks
- `gemma` vs `small-llava`: win_rate=0.500, mean_delta=0.007 across 21 benchmarks
- `small-llava` vs `gemma`: win_rate=0.500, mean_delta=-0.007 across 21 benchmarks
- `small-llava` vs `qwen25-vl`: win_rate=0.214, mean_delta=-0.240 across 21 benchmarks

## Bootstrap Intervals

- `qwen25-vl`: observed=0.519, 90% CI=[0.410, 0.627]
- `gemma`: observed=0.286, 90% CI=[0.184, 0.387]
- `small-llava`: observed=0.279, 90% CI=[0.184, 0.383]

## Telemetry

- `gemma`: mean_wall_clock=12.81s, mean_generation=12.64s, peak_cpu_ram=11.60 GiB, measured=21, estimated=0
- `small-llava`: mean_wall_clock=99.07s, mean_generation=98.97s, peak_cpu_ram=10.58 GiB, measured=21, estimated=0
- `qwen25-vl`: mean_wall_clock=120.03s, mean_generation=119.31s, peak_cpu_ram=12.53 GiB, measured=21, estimated=0

# Comparison Summary

- Best overall mean benchmark-normalized score: `gpt-4.1` (0.758)
- Best raw mean score: `gpt-4.1`
- Most first-place finishes: `qwen25-vl`
- Benchmarks with estimated telemetry: `0` of `105`

## Models

- `gpt-4.1`: mean_score=0.518, normalized=0.758, mean_rank=2.10, wins=10, estimated_stats_benchmarks=0
- `qwen25-vl`: mean_score=0.478, normalized=0.727, mean_rank=1.95, wins=11, estimated_stats_benchmarks=0
- `llava-7b`: mean_score=0.467, normalized=0.661, mean_rank=2.14, wins=7, estimated_stats_benchmarks=0
- `small-llava`: mean_score=0.279, normalized=0.410, mean_rank=3.29, wins=4, estimated_stats_benchmarks=0
- `gemma`: mean_score=0.286, normalized=0.322, mean_rank=3.57, wins=3, estimated_stats_benchmarks=0

## Strongest Family Leaders

- `captioning`: `qwen25-vl` with mean_score=0.325
- `qa`: `gemma` with mean_score=0.600
- `labeling`: `gpt-4.1` with mean_score=0.764
- `detection`: `gpt-4.1` with mean_score=0.080

## Pairwise Edges

- `gpt-4.1` vs `gemma`: win_rate=0.881, mean_delta=0.232 across 21 benchmarks
- `qwen25-vl` vs `small-llava`: win_rate=0.762, mean_delta=0.199 across 21 benchmarks
- `qwen25-vl` vs `gemma`: win_rate=0.762, mean_delta=0.192 across 21 benchmarks
- `llava-7b` vs `gemma`: win_rate=0.714, mean_delta=0.181 across 21 benchmarks
- `gpt-4.1` vs `small-llava`: win_rate=0.667, mean_delta=0.239 across 21 benchmarks

## Bootstrap Intervals

- `gpt-4.1`: observed=0.518, 90% CI=[0.398, 0.637]
- `qwen25-vl`: observed=0.478, 90% CI=[0.365, 0.596]
- `llava-7b`: observed=0.467, 90% CI=[0.338, 0.593]
- `gemma`: observed=0.286, 90% CI=[0.184, 0.387]
- `small-llava`: observed=0.279, 90% CI=[0.180, 0.378]

## Telemetry

- `llava-7b`: mean_wall_clock=1.19s, mean_generation=1.11s, peak_cpu_ram=4.34 GiB, measured=21, estimated=0
- `gpt-4.1`: mean_wall_clock=1.42s, mean_generation=1.35s, peak_cpu_ram=4.54 GiB, measured=21, estimated=0
- `qwen25-vl`: mean_wall_clock=2.70s, mean_generation=2.58s, peak_cpu_ram=4.03 GiB, measured=21, estimated=0
- `gemma`: mean_wall_clock=12.81s, mean_generation=12.64s, peak_cpu_ram=11.60 GiB, measured=21, estimated=0
- `small-llava`: mean_wall_clock=99.07s, mean_generation=98.97s, peak_cpu_ram=10.58 GiB, measured=21, estimated=0

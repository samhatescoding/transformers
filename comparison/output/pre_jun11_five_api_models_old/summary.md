# Comparison Summary

- Best overall mean benchmark-normalized score: `gpt-4.1` (0.820)
- Best raw mean score: `gpt-4.1`
- Most first-place finishes: `gpt-4.1`
- Benchmarks with estimated telemetry: `0` of `60`

## Models

- `gpt-4.1`: mean_score=0.481, normalized=0.820, mean_rank=1.42, wins=8, estimated_stats_benchmarks=0
- `gpt-4o`: mean_score=0.450, normalized=0.755, mean_rank=2.08, wins=5, estimated_stats_benchmarks=0
- `claude-sonnet-4-5`: mean_score=0.322, normalized=0.606, mean_rank=2.83, wins=3, estimated_stats_benchmarks=0
- `gpt-5`: mean_score=0.425, normalized=0.565, mean_rank=2.08, wins=5, estimated_stats_benchmarks=0
- `gemini-3.5-flash`: mean_score=0.017, normalized=0.083, mean_rank=3.83, wins=1, estimated_stats_benchmarks=0

## Strongest Family Leaders

- `qa`: `gpt-4.1` with mean_score=0.867
- `detection`: `claude-sonnet-4-5` with mean_score=0.135

## Pairwise Edges

- `gpt-4.1` vs `gemini-3.5-flash`: win_rate=0.917, mean_delta=0.465 across 12 benchmarks
- `gpt-4o` vs `gemini-3.5-flash`: win_rate=0.875, mean_delta=0.433 across 12 benchmarks
- `claude-sonnet-4-5` vs `gemini-3.5-flash`: win_rate=0.875, mean_delta=0.305 across 12 benchmarks
- `gpt-4o` vs `claude-sonnet-4-5`: win_rate=0.792, mean_delta=0.128 across 12 benchmarks
- `gpt-5` vs `gemini-3.5-flash`: win_rate=0.750, mean_delta=0.408 across 12 benchmarks

## Bootstrap Intervals

- `gpt-4.1`: observed=0.481, 90% CI=[0.288, 0.666]
- `gpt-4o`: observed=0.450, 90% CI=[0.267, 0.639]
- `gpt-5`: observed=0.425, 90% CI=[0.225, 0.625]
- `claude-sonnet-4-5`: observed=0.322, 90% CI=[0.194, 0.454]
- `gemini-3.5-flash`: observed=0.017, 90% CI=[0.000, 0.042]

## Telemetry

- `gpt-4o`: mean_wall_clock=13.32s, mean_generation=5.15s, peak_cpu_ram=1.99 GiB, measured=12, estimated=0
- `claude-sonnet-4-5`: mean_wall_clock=14.98s, mean_generation=3.39s, peak_cpu_ram=1.84 GiB, measured=12, estimated=0
- `gpt-4.1`: mean_wall_clock=15.54s, mean_generation=2.42s, peak_cpu_ram=1.35 GiB, measured=12, estimated=0
- `gpt-5`: mean_wall_clock=16.46s, mean_generation=2.96s, peak_cpu_ram=1.30 GiB, measured=12, estimated=0
- `gemini-3.5-flash`: mean_wall_clock=23.43s, mean_generation=9.01s, peak_cpu_ram=1.56 GiB, measured=12, estimated=0

# Comparison Summary

- Best overall mean benchmark-normalized score: `gpt-4o` (0.803)
- Best raw mean score: `gpt-4o`
- Most first-place finishes: `gpt-4o`
- Benchmarks with estimated telemetry: `0` of `105`

## Models

- `gpt-4o`: mean_score=0.336, normalized=0.803, mean_rank=1.71, wins=12, estimated_stats_benchmarks=0
- `claude-sonnet-4-5`: mean_score=0.239, normalized=0.650, mean_rank=2.52, wins=4, estimated_stats_benchmarks=0
- `gpt-4.1`: mean_score=0.275, normalized=0.469, mean_rank=2.43, wins=8, estimated_stats_benchmarks=0
- `gpt-5`: mean_score=0.243, normalized=0.323, mean_rank=2.81, wins=5, estimated_stats_benchmarks=0
- `gemini-3.5-flash`: mean_score=0.025, normalized=0.157, mean_rank=3.33, wins=2, estimated_stats_benchmarks=0

## Strongest Family Leaders

- `captioning`: `gpt-4o` with mean_score=0.183
- `qa`: `gpt-4.1` with mean_score=0.867
- `detection`: `claude-sonnet-4-5` with mean_score=0.135

## Pairwise Edges

- `gpt-4o` vs `gemini-3.5-flash`: win_rate=0.881, mean_delta=0.311 across 21 benchmarks
- `claude-sonnet-4-5` vs `gemini-3.5-flash`: win_rate=0.833, mean_delta=0.214 across 21 benchmarks
- `gpt-4o` vs `claude-sonnet-4-5`: win_rate=0.833, mean_delta=0.096 across 21 benchmarks
- `gpt-4o` vs `gpt-5`: win_rate=0.714, mean_delta=0.093 across 21 benchmarks
- `claude-sonnet-4-5` vs `gpt-5`: win_rate=0.667, mean_delta=-0.004 across 21 benchmarks

## Bootstrap Intervals

- `gpt-4o`: observed=0.336, 90% CI=[0.221, 0.456]
- `gpt-4.1`: observed=0.275, 90% CI=[0.142, 0.419]
- `gpt-5`: observed=0.243, 90% CI=[0.110, 0.390]
- `claude-sonnet-4-5`: observed=0.239, 90% CI=[0.158, 0.326]
- `gemini-3.5-flash`: observed=0.025, 90% CI=[0.009, 0.044]

## Telemetry

- `gpt-4o`: mean_wall_clock=8.67s, mean_generation=3.81s, peak_cpu_ram=1.99 GiB, measured=21, estimated=0
- `claude-sonnet-4-5`: mean_wall_clock=10.19s, mean_generation=3.38s, peak_cpu_ram=1.86 GiB, measured=21, estimated=0
- `gpt-4.1`: mean_wall_clock=10.74s, mean_generation=2.94s, peak_cpu_ram=1.58 GiB, measured=21, estimated=0
- `gpt-5`: mean_wall_clock=11.17s, mean_generation=3.23s, peak_cpu_ram=1.44 GiB, measured=21, estimated=0
- `gemini-3.5-flash`: mean_wall_clock=19.13s, mean_generation=10.56s, peak_cpu_ram=1.56 GiB, measured=21, estimated=0

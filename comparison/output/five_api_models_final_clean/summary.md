# Comparison Summary

- Best overall mean benchmark-normalized score: `gemini-3.5-flash` (0.790)
- Best raw mean score: `gemini-3.5-flash`
- Most first-place finishes: `gemini-3.5-flash`
- Benchmarks with estimated telemetry: `0` of `220`

## Models

- `gemini-3.5-flash`: mean_score=0.585, normalized=0.790, mean_rank=1.82, wins=24, estimated_stats_benchmarks=0
- `gpt-4.1`: mean_score=0.559, normalized=0.668, mean_rank=2.18, wins=14, estimated_stats_benchmarks=0
- `gpt-4o`: mean_score=0.509, normalized=0.573, mean_rank=2.70, wins=12, estimated_stats_benchmarks=0
- `gpt-5`: mean_score=0.519, normalized=0.567, mean_rank=2.66, wins=11, estimated_stats_benchmarks=0
- `claude-sonnet-4-5`: mean_score=0.419, normalized=0.254, mean_rank=3.82, wins=6, estimated_stats_benchmarks=0

## Strongest Family Leaders

- `captioning`: `gpt-4o` with mean_score=0.170
- `qa`: `gemini-3.5-flash` with mean_score=0.933
- `labeling`: `gemini-3.5-flash` with mean_score=0.738
- `detection`: `gemini-3.5-flash` with mean_score=0.186
- `image_modification_vqa`: `gemini-3.5-flash` with mean_score=0.925
- `image_preference`: `gpt-5` with mean_score=0.750
- `aesthetic_rating`: `gemini-3.5-flash` with mean_score=0.817
- `prompt_reconstruction`: `claude-sonnet-4-5` with mean_score=0.967

## Pairwise Edges

- `gemini-3.5-flash` vs `claude-sonnet-4-5`: win_rate=0.841, mean_delta=0.166 across 44 benchmarks
- `gpt-4.1` vs `claude-sonnet-4-5`: win_rate=0.773, mean_delta=0.140 across 44 benchmarks
- `gpt-4o` vs `claude-sonnet-4-5`: win_rate=0.727, mean_delta=0.089 across 44 benchmarks
- `gpt-5` vs `claude-sonnet-4-5`: win_rate=0.716, mean_delta=0.099 across 44 benchmarks
- `gemini-3.5-flash` vs `gpt-5`: win_rate=0.705, mean_delta=0.066 across 44 benchmarks

## Bootstrap Intervals

- `gemini-3.5-flash`: observed=0.585, 90% CI=[0.496, 0.670]
- `gpt-4.1`: observed=0.559, 90% CI=[0.474, 0.645]
- `gpt-5`: observed=0.519, 90% CI=[0.427, 0.606]
- `gpt-4o`: observed=0.509, 90% CI=[0.424, 0.593]
- `claude-sonnet-4-5`: observed=0.419, 90% CI=[0.340, 0.501]

## Telemetry

- `gpt-4.1`: mean_wall_clock=3.67s, mean_generation=1.85s, peak_cpu_ram=3.16 GiB, measured=44, estimated=0
- `gpt-5`: mean_wall_clock=4.21s, mean_generation=2.10s, peak_cpu_ram=5.69 GiB, measured=44, estimated=0
- `gpt-4o`: mean_wall_clock=5.65s, mean_generation=3.33s, peak_cpu_ram=3.11 GiB, measured=44, estimated=0
- `claude-sonnet-4-5`: mean_wall_clock=6.83s, mean_generation=3.53s, peak_cpu_ram=1.86 GiB, measured=44, estimated=0
- `gemini-3.5-flash`: mean_wall_clock=6.85s, mean_generation=6.55s, peak_cpu_ram=7.27 GiB, measured=44, estimated=0

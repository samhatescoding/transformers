# Comparison Summary

- Best overall mean benchmark-normalized score: `gpt-5.5` (0.825)
- Best raw mean score: `gpt-5.5`
- Most first-place finishes: `qwen2.5-vl-3b-instruct`
- Benchmarks with estimated telemetry: `0` of `294`

## Models

- `gpt-5.5`: mean_score=0.555, normalized=0.825, mean_rank=2.76, wins=8, estimated_stats_benchmarks=0
- `gpt-4.1`: mean_score=0.518, normalized=0.757, mean_rank=3.48, wins=9, estimated_stats_benchmarks=0
- `gpt-5.1`: mean_score=0.508, normalized=0.742, mean_rank=3.76, wins=9, estimated_stats_benchmarks=0
- `gpt-5.4-mini`: mean_score=0.496, normalized=0.727, mean_rank=4.48, wins=7, estimated_stats_benchmarks=0
- `gpt-5.4`: mean_score=0.505, normalized=0.724, mean_rank=4.81, wins=6, estimated_stats_benchmarks=0
- `qwen2.5-vl-3b-instruct`: mean_score=0.478, normalized=0.714, mean_rank=4.81, wins=10, estimated_stats_benchmarks=0
- `gpt-4o`: mean_score=0.495, normalized=0.708, mean_rank=4.62, wins=6, estimated_stats_benchmarks=0
- `gpt-5`: mean_score=0.499, normalized=0.688, mean_rank=4.81, wins=8, estimated_stats_benchmarks=0
- `gpt-5.2`: mean_score=0.480, normalized=0.678, mean_rank=5.19, wins=6, estimated_stats_benchmarks=0
- `llava-1.5-7b-hf`: mean_score=0.467, normalized=0.666, mean_rank=4.57, wins=7, estimated_stats_benchmarks=0
- `gpt-5.3-chat-latest`: mean_score=0.449, normalized=0.632, mean_rank=6.48, wins=6, estimated_stats_benchmarks=0
- `gpt-5.4-nano`: mean_score=0.371, normalized=0.479, mean_rank=9.14, wins=4, estimated_stats_benchmarks=0
- `llava-gemma-2b`: mean_score=0.279, normalized=0.414, mean_rank=8.19, wins=4, estimated_stats_benchmarks=0
- `paligemma-3b-mix-224`: mean_score=0.286, normalized=0.318, mean_rank=10.71, wins=3, estimated_stats_benchmarks=0

## Strongest Family Leaders

- `captioning`: `qwen2.5-vl-3b-instruct` with mean_score=0.325
- `qa`: `paligemma-3b-mix-224` with mean_score=0.600
- `labeling`: `gpt-4.1` with mean_score=0.764
- `detection`: `gpt-5.4` with mean_score=0.159

## Pairwise Edges

- `gpt-5.5` vs `gpt-5.4-nano`: win_rate=0.905, mean_delta=0.184 across 21 benchmarks
- `gpt-5.5` vs `paligemma-3b-mix-224`: win_rate=0.881, mean_delta=0.269 across 21 benchmarks
- `gpt-4.1` vs `paligemma-3b-mix-224`: win_rate=0.881, mean_delta=0.232 across 21 benchmarks
- `gpt-5.1` vs `paligemma-3b-mix-224`: win_rate=0.881, mean_delta=0.222 across 21 benchmarks
- `gpt-5.4` vs `paligemma-3b-mix-224`: win_rate=0.881, mean_delta=0.219 across 21 benchmarks

## Bootstrap Intervals

- `gpt-5.5`: observed=0.555, 90% CI=[0.442, 0.665]
- `gpt-4.1`: observed=0.518, 90% CI=[0.401, 0.639]
- `gpt-5.1`: observed=0.508, 90% CI=[0.394, 0.631]
- `gpt-5.4`: observed=0.505, 90% CI=[0.397, 0.617]
- `gpt-5`: observed=0.499, 90% CI=[0.373, 0.628]
- `gpt-5.4-mini`: observed=0.496, 90% CI=[0.380, 0.615]
- `gpt-4o`: observed=0.495, 90% CI=[0.377, 0.611]
- `gpt-5.2`: observed=0.480, 90% CI=[0.365, 0.597]
- `qwen2.5-vl-3b-instruct`: observed=0.478, 90% CI=[0.363, 0.599]
- `llava-1.5-7b-hf`: observed=0.467, 90% CI=[0.340, 0.594]
- `gpt-5.3-chat-latest`: observed=0.449, 90% CI=[0.333, 0.567]
- `gpt-5.4-nano`: observed=0.371, 90% CI=[0.260, 0.489]
- `paligemma-3b-mix-224`: observed=0.286, 90% CI=[0.188, 0.390]
- `llava-gemma-2b`: observed=0.279, 90% CI=[0.182, 0.381]

## Telemetry

- `gpt-5.4-nano`: mean_wall_clock=1.18s, mean_generation=1.05s, peak_cpu_ram=2.38 GiB, measured=21, estimated=0
- `llava-1.5-7b-hf`: mean_wall_clock=1.19s, mean_generation=1.11s, peak_cpu_ram=4.34 GiB, measured=21, estimated=0
- `gpt-5.4-mini`: mean_wall_clock=1.21s, mean_generation=1.07s, peak_cpu_ram=2.19 GiB, measured=21, estimated=0
- `gpt-5.4`: mean_wall_clock=1.34s, mean_generation=1.21s, peak_cpu_ram=5.04 GiB, measured=21, estimated=0
- `gpt-5.2`: mean_wall_clock=1.41s, mean_generation=1.28s, peak_cpu_ram=2.53 GiB, measured=21, estimated=0
- `gpt-4.1`: mean_wall_clock=1.42s, mean_generation=1.35s, peak_cpu_ram=4.54 GiB, measured=21, estimated=0
- `gpt-5.5`: mean_wall_clock=1.63s, mean_generation=1.50s, peak_cpu_ram=3.55 GiB, measured=21, estimated=0
- `gpt-5.3-chat-latest`: mean_wall_clock=1.87s, mean_generation=1.72s, peak_cpu_ram=3.37 GiB, measured=21, estimated=0
- `gpt-4o`: mean_wall_clock=1.91s, mean_generation=1.77s, peak_cpu_ram=2.56 GiB, measured=21, estimated=0
- `gpt-5.1`: mean_wall_clock=1.94s, mean_generation=1.80s, peak_cpu_ram=3.00 GiB, measured=21, estimated=0
- `gpt-5`: mean_wall_clock=1.97s, mean_generation=1.83s, peak_cpu_ram=4.51 GiB, measured=21, estimated=0
- `qwen2.5-vl-3b-instruct`: mean_wall_clock=2.70s, mean_generation=2.58s, peak_cpu_ram=4.03 GiB, measured=21, estimated=0
- `paligemma-3b-mix-224`: mean_wall_clock=12.81s, mean_generation=12.64s, peak_cpu_ram=11.60 GiB, measured=21, estimated=0
- `llava-gemma-2b`: mean_wall_clock=99.07s, mean_generation=98.97s, peak_cpu_ram=10.58 GiB, measured=21, estimated=0

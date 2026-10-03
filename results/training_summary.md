# Training runs

| run | method | base | r | lr | epochs | trainable_params_M | train_loss | best_eval_loss | train_time_min | peak_gpu_mem_GB |
|---|---|---|---|---|---|---|---|---|---|---|
| dora_r16 | dora | models/Qwen3-8B | 16 | 0.0002 | 2.0 | 45.0 | 0.3423 | 0.244 | 95.0 | 33.5 |
| full | full | models/Qwen3-8B |  | 1e-05 | 2.0 | 8190.7 | 0.3669 | 0.2512 | 34.9 | 37.1 |
| lora_r16 | lora | models/Qwen3-8B | 16 | 0.0002 | 2.0 | 43.6 | 0.3457 | 0.2444 | 46.2 | 28.9 |
| lora_r64 | lora | models/Qwen3-8B | 64 | 0.0001 | 2.0 | 174.6 | 0.335 | 0.2431 | 46.4 | 31.1 |
| qlora_r16 | qlora | models/Qwen3-8B | 16 | 0.0002 | 2.0 | 43.6 | 0.3491 | 0.2482 | 54.7 | 26.4 |

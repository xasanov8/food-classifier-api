| Backend | Batch | Median (ms) | p95 (ms) | Rasm/s |
|---|---|---|---|---|
| torch | 1 | 31.76 | 33.57 | 31.5 |
| torch | 8 | 190.64 | 214.84 | 42.0 |
| onnx | 1 | 19.46 | 22.27 | 51.4 |
| onnx | 8 | 159.52 | 174.01 | 50.2 |

**ONNX tezlashuvi:** batch=1 -> 1.63x, batch=8 -> 1.2x

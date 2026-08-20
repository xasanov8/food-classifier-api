| Backend | Batch | Median (ms) | p95 (ms) | Rasm/s |
|---|---|---|---|---|
| torch | 1 | 31.23 | 41.29 | 32.0 |
| torch | 8 | 196.49 | 255.79 | 40.7 |
| onnx | 1 | 19.40 | 20.69 | 51.5 |
| onnx | 8 | 157.36 | 182.53 | 50.8 |

**ONNX tezlashuvi:** batch=1 -> 1.61x, batch=8 -> 1.25x

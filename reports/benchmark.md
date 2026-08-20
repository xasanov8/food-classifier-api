| Backend | Batch | Median (ms) | p95 (ms) | Rasm/s |
|---|---|---|---|---|
| torch | 1 | 42.34 | 46.00 | 23.6 |
| torch | 8 | 290.49 | 313.54 | 27.5 |
| onnx | 1 | 32.14 | 33.26 | 31.1 |
| onnx | 8 | 251.51 | 263.92 | 31.8 |

**ONNX tezlashuvi:** batch=1 -> 1.32x, batch=8 -> 1.15x

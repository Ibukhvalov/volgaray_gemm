## Simple example of using Rust with objc2-metal for Metal
GeMM: MxK @ KxN = MxN

### Usage:
```bash
cargo build -r
python3 ./scripts/generate_and_run.py <M> <N> <K>
```


## Benchmarks
| Kernel    | Computation time (1024x2048 @ 2048x1024) |
| --------- | ---------------------------------------- |
| Naive     |     21ms                                 |

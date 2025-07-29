## Simple example of using Rust with objc2-metal for Metal
GeMM: MxK @ KxN = MxN

### Usage:
```bash
cargo build -r
python3 ./scripts/generate_and_run.py <M> <N> <K>
```


## Benchmarks
| Kernel       | Computation time (4096) |
| ------------ | ---------------------------------------- |
| Naive        |     950ms                                |
| Block tiling |     280ms                                |

# LowMem-PiPNN

A memory-efficient R32 graph ANN index for squared L2 search. This standalone release packages our V16 main method: PiPNN evidence construction, compact candidate selection and graph replay, followed by FP16 serving with one-bit RaBitQ neighbor routing and FastScan.

## Memory optimizations

- **Packed construction evidence:** B64 evidence uses 7 bytes per slot: a 28-bit neighbor ID, 12-bit cone information and a BF16 value. This bounds intermediate evidence storage while retaining candidates needed by the main construction pipeline.
- **Compact candidate generation:** global 4-bit scores select new forward two-hop and reverse/shared-target candidates (Top64); selected candidates receive exact distance evaluation before graph pruning/replay and R32 completion.
- **Single vector allocation:** input is represented in FP32 during construction; final vertex vectors are converted to FP16 in place. Dead pages and temporary buffers are released after their last use.
- **Compact serving factors:** source-centered one-bit neighbor codes are computed from the original FP32 vectors. Neighbor factors are then stored in FP16 with a shared power-of-two scale to accommodate their range.
- **Contiguous serving pools:** FP16 vertices and per-node routing codes, factors and neighbor IDs use separate contiguous pools. Temporary transpose buffers, caches and thread-local workspaces do not remain part of the serving index.

The B64 construction evidence is distinct from the serving code: serving uses one bit per padded vector dimension, not a fixed 64-bit code for every dimension. The speculative-frontier and learned-projection research variants are not part of this main-method release.

## Build

Requires Linux x86-64, an AVX-512 F/BW/DQ/VL and F16C CPU, C++17 compiler, CMake, Git, OpenMP, Eigen 3.4, and Python 3. NumPy is needed for the small synthetic test. Build on the machine that will run the executables (`-march=native`).

```bash
sudo apt-get install build-essential cmake git libeigen3-dev python3-numpy
git clone https://github.com/Mingle-2012/LowMem-PiPNN.git
cd LowMem-PiPNN
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j16
ctest --test-dir build --output-on-failure
```

CMake fetches pinned Abseil and Highway revisions. No external research checkout or dataset is needed to build or run the synthetic test.

## Run

Matrix files contain two little-endian uint32 values `(number_of_rows, dimension)`, followed by row-major values with no padding. Base/query types are `float32`, `uint8`, or signed `int8`; use `--dtype` and, if different, `--query-dtype`. Ground truth uses the same header and uint32 neighbor IDs, ordered by true squared L2 distance, with at least 10 neighbors per query. No normalization is applied. IDs are zero-based input row numbers. This release supports R32, L2 and fewer than 2^28 base vectors.

```bash
# Build and save PREFIX.meta, PREFIX.vec and PREFIX.blocks.
python3 scripts/run.py build --base data/base.fbin --index indices/deep --dtype float32 --threads 128

# Load the index and report Recall@10 and QPS for each search beam size L.
python3 scripts/run.py search --index indices/deep --query data/query.fbin --gt data/gt.ibin --dtype float32 --L 20,40,60,80,100,150,200,300,400 --rounds 5

# Build and benchmark in the same process, without exporting the index.
python3 scripts/run.py benchmark --base data/base.fbin --query data/query.fbin --gt data/gt.ibin --dtype float32 --threads 128 --rounds 5
```

Use `--dtype uint8` for unsigned byte inputs and `--dtype int8` for signed byte inputs. The wrapper chooses the corresponding executable and sets construction parameters before process startup. Build threads can be reduced for smaller machines; query timing uses a single thread. The C++ loading/search interface is in `include/lowmem_pipnn/index.hpp`.

## Selected large-scale results

These are **selected favorable historical results**, compared specifically with PiPNN + SymphonyQG + RaBitQ, rather than an exhaustive comparison against all methods or datasets. Only Deep100M below contains 100 million vectors; SIFT50M contains 50 million. Values are arithmetic means. Full selected aggregate values and repetition counts are in [selected_results.csv](benchmarks/selected_results.csv).

| Dataset | Method | Build time (s) | Peak build RSS (GiB) | QPS at Recall@10 = 0.80 |
|---|---|---:|---:|---:|
| Deep100M | LowMem-PiPNN | 967.89 | 109.99 | 23,258 |
| Deep100M | PiPNN + SymphonyQG + RaBitQ | 1099.19 | 422.76 | 22,248 |
| SIFT50M | LowMem-PiPNN | 475.43 | 61.14 | 32,066 |
| SIFT50M | PiPNN + SymphonyQG + RaBitQ | 720.79 | 217.01 | 29,769 |
| Deep10M | LowMem-PiPNN | 74.87 | 11.43 | 41,189 |
| Deep10M | PiPNN + SymphonyQG + RaBitQ | 103.58 | 42.18 | 36,648 |
| SIFT10M | LowMem-PiPNN | 81.07 | 12.57 | 47,401 |
| SIFT10M | PiPNN + SymphonyQG + RaBitQ | 176.95 | 43.31 | 33,764 |
| SpaceV10M | LowMem-PiPNN | 126.26 | 11.62 | 38,593 |
| SpaceV10M | PiPNN + SymphonyQG + RaBitQ | 195.42 | 42.45 | 32,458 |

Historical protocol: AMD EPYC 9654, native AVX-512; construction uses 128 threads, search one thread, BLAS one thread; degree 32, seed 12345 and root subtree batch 32. Search L is swept over 20,40,60,80,100,150,200,300,400, using the original full queries and ground truth. QPS at recall 0.80 is interpolated between adjacent L points **within each build/search round**, without extrapolation, then averaged. It is not the throughput of a single fixed-L operating point.

LowMem-PiPNN uses five independent builds and five search rounds per build; the historical comparator uses three builds and three rounds. These are separate historical cohorts, with potentially different CPU placement, rather than paired timing trials. Build time starts with resident input and ends at a ready index, excluding input loading and index export. Peak build RSS combines process high-water measurements and a 100 ms external sampler, stopping before query loading. The measurements predate standalone packaging; the release is validated by small synthetic build/save/load/search tests rather than a fresh 100M rerun.

## License

Original contributions are released under the [MIT License](LICENSE). Retained or adapted third-party code keeps its original MIT or Apache-2.0 license; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

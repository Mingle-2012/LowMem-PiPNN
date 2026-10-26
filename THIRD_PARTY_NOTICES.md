# Third-party notices

The root MIT license covers original LowMem-PiPNN contributions. Existing upstream code and adapted portions retain their original licenses and copyright notices.

- PiPNN: https://github.com/ParAlg/PiPNN — MIT; required graph construction headers only. See `third_party/pipnn/LICENSE`.
- ParlayLib: https://github.com/cmuparlay/parlaylib — MIT; required headers from commit `7cdb4cae8f020525f5eb4ad82e2565d1e38cfbc3`. See `third_party/parlay/LICENSE`.
- RaBitQ Library / SymphonyQG: https://github.com/VectorDB-NTU/RaBitQ-Library — Apache-2.0. See `third_party/rabitq/LICENSE`. This distribution contains the required quantization, rotation, FastScan and SIMD utilities. The unused graph builders and HNSW/IVF implementations are excluded; dispatch has been reduced accordingly. The quantized-query helper was extracted from the upstream QG implementation. `include/lowmem_pipnn/index.hpp` adapts its serving/search logic and retains Apache-2.0 licensing.
- Abseil: https://github.com/abseil/abseil-cpp — Apache-2.0, fetched by CMake at commit `bc257a88f7c1939f24e0379f14a3589e926c950c`.
- Highway: https://github.com/google/highway — Apache-2.0, fetched by CMake at commit `9d1374260ad6f234aa7bafcfccf48abe96c41999`.
- Eigen 3.4: https://eigen.tuxfamily.org — system dependency; MPL-2.0 and applicable upstream per-file licenses. Eigen implementation headers are not bundled here.

Modifications include compact evidence storage, graph replay, candidate selection, FP16 vector/factor storage, buffer lifetime management, standalone build/search interfaces and removal of unused research branches. Upstream notices within retained files remain applicable.

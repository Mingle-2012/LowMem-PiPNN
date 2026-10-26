#pragma once

static void main_v16_search(const lowmem::Index& idx) {
    const char* query_path = std::getenv("LOWMEM_QUERY");
    const char* gt_path = std::getenv("LOWMEM_GT");
    if (!query_path || !gt_path) throw std::runtime_error("LOWMEM_QUERY/LOWMEM_GT required");
    const bool preflight = std::getenv("LOWMEM_PREFLIGHT");
    if (preflight) {
        size_t invalid = 0;
        for (size_t u = 0; u < idx.m.n; ++u)
            for (size_t k = 0; k < 64; ++k) {
                uint16_t bits;
                std::memcpy(&bits, idx.block(u) + idx.cb + k * 2, 2);
                invalid += (bits & 0x7c00u) == 0x7c00u;
            }
        std::cout << "FACTOR_FINITE checked=" << idx.m.n * 64 << " invalid=" << invalid << '\n' << std::flush;
        if (invalid) throw std::runtime_error("FP16 correction factor overflow/nonfinite");
    }
    uint32_t shape[2];
    { std::ifstream f(query_path, std::ios::binary); f.read(reinterpret_cast<char*>(shape), 8);
      if (!f) throw std::runtime_error("query header"); }
    const bool full_queries = !preflight || std::getenv("LOWMEM_FULL_QUERY_PREFLIGHT");
    size_t limit = full_queries ? 0 : std::min<size_t>(128, shape[0]);
    auto queries = load_matrix(query_path, limit,
        std::getenv("LOWMEM_QUERY_DTYPE") ? std::getenv("LOWMEM_QUERY_DTYPE") : "float32");
    auto truth = load_truth(gt_path, queries.n);
    if (queries.d != idx.m.d || truth.size() != size_t(queries.n) * 10)
        throw std::runtime_error("full query/GT dimensions mismatch");
    if (full_queries && std::any_of(truth.begin(), truth.end(), [&](uint32_t id) { return id >= idx.m.n; }))
        throw std::runtime_error("GT ID outside formal dataset");
    int rounds = std::getenv("LOWMEM_SEARCH_ROUNDS") ? std::stoi(std::getenv("LOWMEM_SEARCH_ROUNDS")) : 5;
    if (rounds <= 0) throw std::runtime_error("positive search rounds required");
    std::string values = std::getenv("LOWMEM_SEARCH_GRID") ? std::getenv("LOWMEM_SEARCH_GRID") : "20,40,60,80,100,150,200,300,400";
    lowmem::Pin pin;
    std::vector<uint32_t> result(size_t(queries.n) * 10);
    float distances[10];
    std::cout << "SEARCH_PROTOCOL full_queries=" << full_queries << " nq=" << queries.n
              << " rounds=" << rounds << " warmups=1 threads=1 cpu=" << pin.cpu
              << " metric=" << (idx.m.ip ? "raw_MIPS_1-dot" : "raw_L2")
              << " preflight_not_formal_result=" << preflight << '\n' << std::flush;
    for (size_t ef : grid(values)) {
        lowmem::Scratch scratch(idx.m.n, idx.m.p, ef);
        for (int round = -1; round < rounds; ++round) {
            auto start = Clock::now();
            for (size_t i = 0; i < queries.n; ++i)
                lowmem::search(idx, queries.x.data() + i * queries.d, result.data() + i * 10, distances, scratch);
            double seconds = elapsed(start);
            if (round < 0) continue;
            size_t hits = 0;
            for (size_t i = 0; i < queries.n; ++i)
                for (size_t k = 0; k < 10; ++k)
                    hits += std::find(result.data() + i * 10, result.data() + i * 10 + 10,
                                      truth[i * 10 + k]) != result.data() + i * 10 + 10;
            std::cout << "SEARCH method=main_v16 ef=" << ef << " round=" << round
                      << " qps=" << queries.n / seconds << " recall=" << double(hits) / truth.size()
                      << " nq=" << queries.n << " threads=1 seconds=" << seconds << '\n' << std::flush;
        }
    }
    std::cout << "PRIMARY_DONE builds=1 search_rounds_per_L=" << rounds << " index_writes=0\n" << std::flush;
}

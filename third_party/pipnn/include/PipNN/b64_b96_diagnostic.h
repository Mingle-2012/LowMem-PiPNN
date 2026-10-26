#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>
#include "../../../parlay/include/parlay/parallel.h"

namespace parlayANN {

// Raw N-byte u8 degree files, exactly one entry per source, no header.
class B64B96FinalDegrees {
 public:
  explicit B64B96FinalDegrees(size_t n) : output_(n, 0) {
    if (const char* path = std::getenv("PIPNN_B64_FINAL_DEGREES_IN")) {
      caps_.resize(n);
      std::ifstream in(path, std::ios::binary);
      if (!in || !in.read(reinterpret_cast<char*>(caps_.data()), n) || in.peek() != EOF)
        throw std::runtime_error("B64 reference degrees must contain exactly N u8 entries");
      for (uint8_t cap : caps_)
        if (cap > 64) throw std::runtime_error("B64 reference degree exceeds R64");
    }
  }
  size_t limit(size_t i) const { return caps_.empty() ? 64 : caps_[i]; }
  void record(size_t i, size_t degree) {
    if (degree > limit(i)) throw std::runtime_error("diagnostic exceeds per-node B64 degree");
    output_[i] = static_cast<uint8_t>(degree);
  }
  void finish() const {
    uint64_t edges = 0, cap_edges = 0, short_rows = 0;
    for (size_t i = 0; i < output_.size(); ++i) {
      edges += output_[i];
      if (!caps_.empty()) {
        cap_edges += caps_[i];
        short_rows += output_[i] < caps_[i];
      }
    }
    std::cout << "B64_B96_FINAL total_edges=" << edges
              << " average_degree=" << (output_.empty() ? 0 : double(edges) / output_.size())
              << " degree_matched=" << !caps_.empty()
              << " reference_edges=" << cap_edges << " below_cap_rows=" << short_rows << '\n';
    if (const char* path = std::getenv("PIPNN_B64_FINAL_DEGREES_OUT")) {
      std::ofstream out(path, std::ios::binary);
      if (!out || !out.write(reinterpret_cast<const char*>(output_.data()), output_.size()))
        throw std::runtime_error("cannot export B64/B96 final degrees");
    }
  }
 private:
  std::vector<uint8_t> caps_, output_;
};

template <class GraphH>
void b64_b96_write_candidate_digest(GraphH& graph, size_t n, const char* path) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("cannot open B64/B96 candidate digest file");
  constexpr size_t chunk = 1000000;
  std::vector<uint64_t> digests(std::min(n, chunk));
  for (size_t begin = 0; begin < n; begin += chunk) {
    const size_t count = std::min(chunk, n - begin);
    parlay::parallel_for(0, count, [&](size_t j) {
      thread_local std::vector<std::tuple<uint32_t, uint16_t, uint32_t>> keys;
      keys.clear();
      for (auto ngh : graph.Neighbors(begin + j)) {
        uint32_t norm_bits;
        std::memcpy(&norm_bits, &ngh.norm, sizeof(norm_bits));
        keys.emplace_back(ngh.point_id, ngh.hash, norm_bits);
      }
      std::sort(keys.begin(), keys.end());
      uint64_t h = 1469598103934665603ULL;
      for (auto [id, cone, norm] : keys) {
        h = (h ^ id) * 1099511628211ULL;
        h = (h ^ cone) * 1099511628211ULL;
        h = (h ^ norm) * 1099511628211ULL;
      }
      digests[j] = (h ^ keys.size()) * 1099511628211ULL;
    });
    if (!out.write(reinterpret_cast<const char*>(digests.data()), count * sizeof(uint64_t)))
      throw std::runtime_error("cannot write B64/B96 candidate digests");
  }
}

}  // namespace parlayANN

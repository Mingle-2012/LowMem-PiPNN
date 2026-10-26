#pragma once
// Production candidate-stream reducer. Experimental adaptive/B96/oracle
// implementations remain in lowmem_twowitness_graph.h, not in this file.
#include <absl/synchronization/mutex.h>
#include <Eigen/Core>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <malloc.h>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
#include "../../third_party/pipnn/include/utils/graph.h"
#include "../../third_party/pipnn/include/PipNN/aea_total_memory_planner.h"
#include "../../third_party/pipnn/include/PipNN/lowmem_graph_workspace.h"
#include "../../third_party/parlay/include/parlay/parallel.h"

namespace parlayANN {

struct __attribute__((packed)) B64BFloat16 {
  uint16_t bits = 0;
  B64BFloat16() = default;
  B64BFloat16(float value) {
    Eigen::bfloat16 encoded(value);
    std::memcpy(&bits, &encoded, sizeof(bits));
  }
  operator float() const {
    Eigen::bfloat16 encoded;
    std::memcpy(&encoded, &bits, sizeof(bits));
    return static_cast<float>(encoded);
  }
};

struct __attribute__((packed)) B64Candidate {
  uint8_t key[5];  // Exact 28-bit ID + 12-bit cone, little-endian.
  B64BFloat16 norm;
};
static_assert(sizeof(B64BFloat16) == 2 && sizeof(B64Candidate) == 7,
              "B64 candidate layout must remain seven bytes");

// One nearest stored representative per cone, at most 64 representatives.
// Rows are cone-sorted. Incoming batches MUST be distance-sorted, as in PiPNN.
// Preserve native FP32-vs-decoded-BF16 comparisons, strict inequalities and
// first maximum tie rule; this is a refactor, not an alternative pruning rule.
template <typename indexType_, typename distanceType_, typename sketchHash>
class UniformB64Graph {
 public:
  using indexType = indexType_;
  using distanceType = distanceType_;
  using T = B64Candidate;
  static constexpr size_t kCapacity = 64;
  static constexpr size_t kFinalStride = 32;
  static constexpr size_t kLockStripes = 1u << 16;
  static constexpr uint64_t kIdMask = (uint64_t{1} << 28) - 1;

  UniformB64Graph(size_t n, sketchHash& hash, const AEAMemoryPlan& plan)
      : n_(n), hash_(hash), degrees_(n, 0), max_index_(n, 0), max_value_(n),
        locks_(kLockStripes) {
    if (plan.evidence_floor != kCapacity || plan.mandatory_floor_bytes ||
        plan.overflow_bytes)
      throw std::runtime_error("main B64 requires floor64 and zero overflow budget");
    for (const char* key : {"PIPNN_AEA_TIERS_IN", "PIPNN_AEA_TIERS_OUT",
                           "PIPNN_B64_B96_CHECKPOINT_IN", "PIPNN_B64_B96_CHECKPOINT_OUT"})
      if (std::getenv(key))
        throw std::runtime_error("main B64 does not support experimental tiers/checkpoints");
  }
  ~UniformB64Graph() {
    std::free(base_);
    if (final_rows_) munmap(final_rows_, n_ * kFinalStride * sizeof(indexType));
  }
  UniformB64Graph(const UniformB64Graph&) = delete;
  UniformB64Graph& operator=(const UniformB64Graph&) = delete;

  size_t size() const { return n_; }
  size_t degree(size_t u) const { return degrees_[u]; }
  size_t capacity(size_t) const { return kCapacity; }

  void allocate_graph(size_t n) {
    if (n != n_) throw std::runtime_error("B64 graph size mismatch");
    if (base_) return;
    base_ = static_cast<T*>(std::malloc(n_ * kCapacity * sizeof(T)));
    if (!base_) throw std::bad_alloc();
    std::cout << "B64_FIXED_STORAGE capacity=64 slot_bytes=7 base_bytes="
              << n_ * kCapacity * sizeof(T) << " overflow_bytes=0\n";
  }
  // Compatibility with the shared root-batch driver: fixed B64 never promotes.
  void allocate_stage(size_t) {}

  template <typename Range>
  void merge_neighbors(size_t u, const Range& candidates) {
    absl::MutexLock guard(&locks_[u & (kLockStripes - 1)]);
    size_t d = degrees_[u];
    float maxv = d ? static_cast<float>(max_value_[u]) : 0.0f;
    for (const auto& candidate : candidates) {
      const indexType v = candidate.first;
      const float distance = candidate.second;
      if (v == u) continue;
      if (d == kCapacity && distance >= maxv) break;
      const uint16_t cone = static_cast<uint16_t>(hash_.relative_hash(u, v));
      size_t lo = 0, hi = d;
      while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (stored_cone(at(u, mid)) < cone) lo = mid + 1;
        else hi = mid;
      }
      size_t pos = lo;
      if (pos < d && stored_cone(at(u, pos)) == cone) {
        if (static_cast<float>(at(u, pos).norm) > distance) {
          at(u, pos) = make(v, cone, distance);
          if (pos == max_index_[u]) recompute_max(u, d);
          maxv = static_cast<float>(max_value_[u]);
        }
        continue;
      }
      if (d < kCapacity) {
        for (size_t j = d; j > pos; --j) at(u, j) = at(u, j - 1);
        at(u, pos) = make(v, cone, distance);
        if (d == 0 || distance > maxv) {
          max_index_[u] = static_cast<uint8_t>(pos);
          max_value_[u] = B64BFloat16(distance);
          maxv = static_cast<float>(max_value_[u]);
        } else if (pos <= max_index_[u]) ++max_index_[u];
        ++d;
      } else {
        size_t worst = max_index_[u];
        if (worst < pos) {
          for (size_t j = worst; j + 1 < pos; ++j) at(u, j) = at(u, j + 1);
          --pos;
        } else if (pos < worst) {
          for (size_t j = worst; j > pos; --j) at(u, j) = at(u, j - 1);
        }
        at(u, pos) = make(v, cone, distance);
        recompute_max(u, d);
        maxv = static_cast<float>(max_value_[u]);
      }
    }
    degrees_[u] = static_cast<uint8_t>(d);
  }


  class NeighborRange {
   public:
    struct value_type { indexType point_id; float norm; uint16_t hash; };
    struct iterator {
      const UniformB64Graph* graph; size_t node, pos;
      value_type operator*() const {
        const T& candidate = graph->at(node, pos);
        return {stored_id(candidate), static_cast<float>(candidate.norm),
                stored_cone(candidate)};
      }
      iterator& operator++() { ++pos; return *this; }
      bool operator!=(const iterator& other) const { return pos != other.pos; }
    };
    NeighborRange(const UniformB64Graph* graph, size_t node, size_t degree)
        : graph_(graph), node_(node), degree_(degree) {}
    iterator begin() const { return {graph_, node_, 0}; }
    iterator end() const { return {graph_, node_, degree_}; }
    size_t size() const { return degree_; }
   private:
    const UniformB64Graph* graph_; size_t node_, degree_;
  };
  NeighborRange Neighbors(size_t u) const { return {this, u, degrees_[u]}; }



  // Construction-only final staging capacity matches the fixed output R32; candidate capacity remains64.
  // Only completed candidate rows may be discarded; no intermediate index IO.
  void finalize() {
    if (final_rows_) throw std::runtime_error("B64 finalized twice");
    const size_t bytes = n_ * kFinalStride * sizeof(indexType);
    final_rows_ = static_cast<indexType*>(mmap(nullptr, bytes,
        PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0));
    if (final_rows_ == MAP_FAILED) { final_rows_ = nullptr; throw std::bad_alloc(); }
    uint64_t count = 0;
    for (auto degree : degrees_) count += degree;
    std::cout << "B64_CANDIDATES total=" << count << " average="
              << (n_ ? double(count) / n_ : 0) << " capacity=64\n"
              << "B64_FIXED_STORAGE metadata_live_bytes=" << metadata_bytes()
              << " candidate_live_bytes=" << candidate_live_bytes() << '\n';
  }
  template <class Range>
  void write_final_row_r64(size_t u, const Range& ids) {
    if (!final_rows_ || ids.size() > kFinalStride)
      throw std::runtime_error("B64 final row unavailable or too large");
    size_t j = 0;
    for (auto v : ids) final_rows_[u * kFinalStride + j++] = static_cast<indexType>(v);
    degrees_[u] = static_cast<uint8_t>(j);
  }
  void discard_candidate_rows(size_t begin, size_t end) {
    if (!base_ || end <= begin) return;
    const size_t page = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
    size_t relative_begin=begin,relative_end=end;
    uintptr_t lo = (reinterpret_cast<uintptr_t>(base_) + relative_begin * kCapacity * sizeof(T) + page - 1) & ~(page - 1);
    uintptr_t hi = (reinterpret_cast<uintptr_t>(base_) + relative_end * kCapacity * sizeof(T)) & ~(page - 1);
    if (hi > lo) madvise(reinterpret_cast<void*>(lo), hi - lo, MADV_DONTNEED);
  }
  void handoff_r64_streaming(Graph<indexType>& out) {
    if (!final_rows_) throw std::runtime_error("B64 final adjacency missing");
    auto* degrees = static_cast<indexType*>(std::malloc(n_ * sizeof(indexType)));
    if (!degrees) throw std::bad_alloc();
    parlay::parallel_for(0, n_, [&](size_t u) { degrees[u] = degrees_[u]; });
    out.adopt_mmap_packed_graph(final_rows_, degrees, kFinalStride, n_,
                               n_ * kFinalStride * sizeof(indexType));
    final_rows_ = nullptr;
    std::free(base_); base_ = nullptr;
    std::vector<uint8_t>().swap(degrees_);
    std::vector<uint8_t>().swap(max_index_);
    std::vector<B64BFloat16>().swap(max_value_);
    std::vector<absl::Mutex>().swap(locks_);
    malloc_trim(0);
  }
  void release_hash_state() {
    hash_.release_sketch_data(); malloc_trim(0);
  }
  size_t candidate_live_bytes() const {
    return (base_ ? n_ * kCapacity * sizeof(T) : 0) + metadata_bytes();
  }
  void report_budget_phase(const char* phase) const {
    size_t workspace = 0;
    for (const auto& worker : pipnn_clean_workspaces())
      workspace += (worker.points.capacity() + worker.distances.capacity()) * sizeof(float);
    std::cout << "AEA_BUDGET_PHASE phase=" << phase << " base_candidate_bytes="
              << (base_ ? n_ * kCapacity * sizeof(T) : 0)
              << " metadata_bytes=" << metadata_bytes()
              << " rbc_leaf_workspace_bytes=" << workspace
              << " overflow_live_bytes=0 overflow_reserved_bytes=0\n";
    pipnn_clean_report_memory(phase);
  }

 private:
  T& at(size_t u, size_t j) { return base_[row_index(u) * kCapacity + j]; }
  const T& at(size_t u, size_t j) const { return base_[row_index(u) * kCapacity + j]; }
  size_t row_index(size_t u)const{
    return u;
  }
  static indexType stored_id(const T& candidate) {
    return static_cast<indexType>(uint32_t(candidate.key[0]) |
        (uint32_t(candidate.key[1]) << 8) | (uint32_t(candidate.key[2]) << 16) |
        ((uint32_t(candidate.key[3]) & 0x0fU) << 24));
  }
  static uint16_t stored_cone(const T& candidate) {
    return static_cast<uint16_t>((candidate.key[3] >> 4) |
                                 (uint16_t(candidate.key[4]) << 4));
  }
  static T make(indexType v, uint16_t cone, float distance) {
    if (uint64_t(v) > kIdMask || cone >= (1u << 12))
      throw std::runtime_error("B64 key exceeds exact encoding");
    const uint64_t value = uint64_t(v) | (uint64_t(cone) << 28);
    T candidate{};
    for (unsigned b = 0; b < 5; ++b) candidate.key[b] = uint8_t(value >> (8 * b));
    candidate.norm = B64BFloat16(distance);
    return candidate;
  }
  void recompute_max(size_t u, size_t d) {
    size_t index = 0;
    float value = d ? static_cast<float>(at(u, 0).norm) : 0.0f;
    for (size_t j = 1; j < d; ++j) {
      float next = static_cast<float>(at(u, j).norm);
      if (next > value) { value = next; index = j; }
    }
    max_index_[u] = static_cast<uint8_t>(index);
    max_value_[u] = B64BFloat16(value);
  }
  size_t metadata_bytes() const {
    return degrees_.capacity() + max_index_.capacity() +
           max_value_.capacity() * sizeof(B64BFloat16) +
           locks_.capacity() * sizeof(absl::Mutex);
  }
  size_t n_;
  sketchHash& hash_;
  T* base_ = nullptr;
  indexType* final_rows_ = nullptr;
  std::vector<uint8_t> degrees_, max_index_;
  std::vector<B64BFloat16> max_value_;
  std::vector<absl::Mutex> locks_;
};
}  // namespace parlayANN

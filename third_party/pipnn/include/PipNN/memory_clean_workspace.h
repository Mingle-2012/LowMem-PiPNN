#pragma once

// Semantics-preserving construction workspace shared by PiPNN-Clean and the
// isolated low-memory Two-Witness variant.  Upstream PiPNN keeps its original
// fixed 2048x2048 thread-local buffers for artifact reproduction.

#include <algorithm>
#include <fstream>
#include <malloc.h>
#include <string>
#include <stdexcept>
#include <sys/resource.h>
#include <vector>

#include "../../../parlay/include/parlay/parallel.h"
#include "../../../parlay/include/parlay/alloc.h"

namespace parlayANN {

struct PipNNMemoryCleanWorkspace {
  // Routing and leaf kernels never execute simultaneously on one Parlay
  // worker. Reuse the same two buffers across phases instead of retaining two
  // independent high-water capacities per worker.
  std::vector<float> points, distances;
};

struct PipNNWorkspaceLimits {
  size_t point_floats = 0;
  size_t distance_floats = 0;
};

inline PipNNWorkspaceLimits& pipnn_clean_workspace_limits() {
  static PipNNWorkspaceLimits limits;
  return limits;
}

inline void pipnn_clean_configure_workspace(size_t dim, size_t leaf_max,
                                            size_t max_leaders) {
  auto& limits = pipnn_clean_workspace_limits();
  limits.point_floats = std::max(size_t{256} * dim, leaf_max * dim);
  limits.distance_floats = std::max(size_t{256} * max_leaders,
                                    size_t{128} * 128);
}

inline float* pipnn_clean_bounded_buffer(std::vector<float>& buffer,
                                          size_t requested, size_t limit) {
  if (limit && requested > limit)
    throw std::runtime_error("PiPNN workspace request exceeds planned capacity");
  if (limit && buffer.capacity() < limit) buffer.reserve(limit);
  if (buffer.size() < requested) buffer.resize(requested);
  return buffer.data();
}

inline std::vector<PipNNMemoryCleanWorkspace>& pipnn_clean_workspaces() {
  static std::vector<PipNNMemoryCleanWorkspace> pools(parlay::num_workers());
  return pools;
}

inline float* pipnn_clean_routing_points(size_t n) {
  auto& v = pipnn_clean_workspaces()[parlay::worker_id()].points;
  return pipnn_clean_bounded_buffer(v, n,
      pipnn_clean_workspace_limits().point_floats);
}

inline float* pipnn_clean_routing_distances(size_t n) {
  auto& v = pipnn_clean_workspaces()[parlay::worker_id()].distances;
  return pipnn_clean_bounded_buffer(v, n,
      pipnn_clean_workspace_limits().distance_floats);
}

inline float* pipnn_clean_leaf_points(size_t n) {
  auto& v = pipnn_clean_workspaces()[parlay::worker_id()].points;
  return pipnn_clean_bounded_buffer(v, n,
      pipnn_clean_workspace_limits().point_floats);
}

inline float* pipnn_clean_leaf_distances(size_t n) {
  auto& v = pipnn_clean_workspaces()[parlay::worker_id()].distances;
  return pipnn_clean_bounded_buffer(v, n,
      pipnn_clean_workspace_limits().distance_floats);
}

inline void pipnn_clean_release_routing_workspaces() {
  // In the streaming DFS pipeline leaf work is interleaved with routing and
  // shares this storage. It is released once both activities finish.
  for (auto& w : pipnn_clean_workspaces()) {
    std::vector<float>().swap(w.points);
    std::vector<float>().swap(w.distances);
  }
  malloc_trim(0);
}

inline void pipnn_clean_release_leaf_workspaces() {
  for (auto& w : pipnn_clean_workspaces()) {
    std::vector<float>().swap(w.points);
    std::vector<float>().swap(w.distances);
  }
  malloc_trim(0);
}

inline void pipnn_clean_trim_allocator() {
  parlay::internal::memory_clear();
  malloc_trim(0);
}

inline void pipnn_clean_report_memory(const char* phase) {
  std::ifstream in("/proc/self/status");
  std::string line, rss, hwm;
  while (std::getline(in, line)) {
    if (line.compare(0, 6, "VmRSS:") == 0) rss = line.substr(6);
    if (line.compare(0, 6, "VmHWM:") == 0) hwm = line.substr(6);
  }
  struct rusage usage {};
  getrusage(RUSAGE_SELF, &usage);
  std::cout << "pipnn_clean_phase_memory phase=" << phase
            << " rss=" << rss << " hwm=" << hwm
            << " getrusage_maxrss_kib=" << usage.ru_maxrss << '\n';
}

inline void pipnn_clean_report_workspace_bytes(const char* phase) {
  size_t routing_points = 0, routing_distances = 0;
  size_t leaf_points = 0, leaf_distances = 0;
  for (const auto& w : pipnn_clean_workspaces()) {
    routing_points += w.points.capacity() * sizeof(float);
    routing_distances += w.distances.capacity() * sizeof(float);
  }
  std::cout << "pipnn_clean_workspace_bytes phase=" << phase
            << " routing_points=" << routing_points
            << " routing_distances=" << routing_distances
            << " leaf_points=" << leaf_points
            << " leaf_distances=" << leaf_distances << '\n';
}

inline void pipnn_clean_report_parlay_allocator(const char* phase) {
  const auto [used, reserved] = parlay::internal::memory_usage();
  std::cout << "pipnn_clean_parlay_bytes phase=" << phase
            << " used=" << used << " reserved=" << reserved << '\n';
}

}  // namespace parlayANN

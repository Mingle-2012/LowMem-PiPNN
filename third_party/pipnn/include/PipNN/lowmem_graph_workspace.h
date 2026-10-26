#pragma once
#include "memory_clean_workspace.h"

namespace parlayANN {
inline float* lowmem_routing_points(size_t n) { return pipnn_clean_routing_points(n); }
inline float* lowmem_routing_distances(size_t n) { return pipnn_clean_routing_distances(n); }
inline float* lowmem_leaf_points(size_t n) { return pipnn_clean_leaf_points(n); }
inline float* lowmem_leaf_distances(size_t n) { return pipnn_clean_leaf_distances(n); }
inline void lowmem_release_routing_workspaces() { pipnn_clean_release_routing_workspaces(); }
inline void lowmem_release_leaf_workspaces() { pipnn_clean_release_leaf_workspaces(); }
inline void lowmem_trim_allocator() { pipnn_clean_trim_allocator(); }
inline void lowmem_report_memory(const char* phase) { pipnn_clean_report_memory(phase); }
}  // namespace parlayANN

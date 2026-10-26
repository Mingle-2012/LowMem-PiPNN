#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace parlayANN {

struct AEAMemoryPlan {
  uint64_t total_bytes = 0;
  uint64_t fixed_bytes = 0;
  uint64_t routing_bytes = 0;
  uint64_t leaf_bytes = 0;
  uint64_t allocation_bytes = 0;
  uint64_t final_bytes = 0;
  uint64_t serialization_bytes = 0;
  uint64_t root_merge_copy_bytes = 0;
  uint64_t workspace_bytes = 0;
  uint64_t prehash_transient_bytes = 0;
  uint64_t base_floor_bytes = 0;
  uint64_t mandatory_floor_bytes = 0;
  uint64_t mandatory_candidate_bytes = 0;
  uint64_t minimum_total_bytes = 0;
  uint64_t guard_bytes = 0;
  uint64_t overflow_bytes = 0;
  uint32_t evidence_floor = 32;
  uint32_t maximum_logical_capacity = 128;
  bool total_interface = false;
};

inline uint64_t aea_parse_mib(const char* value, const char* name) {
  if (!value || !*value) throw std::runtime_error(std::string(name) + " is empty");
  char* end = nullptr;
  const uint64_t mib = std::strtoull(value, &end, 10);
  if (end == value || *end != '\0' || mib > (UINT64_MAX >> 20))
    throw std::runtime_error(std::string(name) + " must be an integer MiB");
  return mib << 20;
}

inline AEAMemoryPlan make_aea_memory_plan(
    uint64_t n, uint64_t dim, uint64_t element_bytes,
    uint64_t aligned_vector_bytes, uint64_t threads,
    uint64_t base_capacity, uint64_t candidate_bytes,
    uint64_t leaf_max, uint64_t max_leaders,
    uint64_t root_fanout, uint64_t hash_bits,
    uint64_t final_degree, uint64_t final_extra_per_node = 0,
    bool emit = true) {
  constexpr uint64_t MiB = 1ull << 20;
  AEAMemoryPlan plan;
  if (base_capacity != 64)
    throw std::runtime_error("B64/B96 planner requires contiguous base64");
  plan.evidence_floor = 64;
  plan.maximum_logical_capacity = 96;
  if (std::getenv("PIPNN_AEA_TIERS_IN"))
    throw std::runtime_error("B64/B96 diagnostic uses online gain64 only");
  const char* total = std::getenv("PIPNN_AEA_TOTAL_MEMORY_MIB");
  const char* debug = std::getenv("PIPNN_AEA_OVERFLOW_MIB");
  if (total && debug)
    throw std::runtime_error("set total memory budget or debug overflow, not both");
  if (!total && !debug && !std::getenv("PIPNN_AEA_TIERS_IN"))
    throw std::runtime_error("AEA requires PIPNN_AEA_TOTAL_MEMORY_MIB");
  plan.total_interface = total != nullptr;
  if (total) plan.total_bytes = aea_parse_mib(total, "PIPNN_AEA_TOTAL_MEMORY_MIB");
  if (const char* floor = std::getenv("PIPNN_AEA_EVIDENCE_FLOOR")) {
    char* end = nullptr;
    const unsigned long value = std::strtoul(floor, &end, 10);
    if (end == floor || *end != '\0' || (value != 32 && value != 64))
      throw std::runtime_error("PIPNN_AEA_EVIDENCE_FLOOR must be 32 or 64");
    plan.evidence_floor = static_cast<uint32_t>(value);
  }
  if (plan.evidence_floor != 64)
    throw std::runtime_error("B64/B96 diagnostic forbids evidence floor below64");

  // Mandatory row metadata: degree, max index, bf16 max, offset, tier,
  // two marginal signals, and two 64-bit distinct-cone sketches.
  const uint64_t metadata = n * (
      // One overflow offset; gain64 and the unchanged distinct-cone sketches.
      // Reserve two bytes/node for degree cap and degree export in every A/B/C
      // plan, so B and C get identical overflow budgets at the same target.
      1 + 1 + 2 + 4 + 2 + 8 + 8 + 2
      ) +
                            (1ull << 16) * 8;
  const uint64_t vectors = n * aligned_vector_bytes;
  const uint64_t base = n * base_capacity * candidate_bytes;
  plan.mandatory_candidate_bytes = base;
  plan.fixed_bytes = vectors + base + metadata;
  const uint64_t hash_state = n * hash_bits * sizeof(float) +
                              hash_bits * dim * sizeof(float);
  const uint64_t workspace_points =
      std::max<uint64_t>(256 * dim, leaf_max * dim) * sizeof(float) * threads;
  const uint64_t workspace_distances =
      std::max<uint64_t>(256 * max_leaders, 128 * 128) * sizeof(float) * threads;
  const uint64_t worker_workspace = workspace_points + workspace_distances;
  // The current leaf path keeps one forward and one reverse 10-byte proposal
  // on each worker's stack; it does not materialize a proposal array.
  plan.prehash_transient_bytes = 2 * 10 * threads;
  plan.workspace_bytes = worker_workspace + plan.prehash_transient_bytes;
  // At root routing, the 8-byte source/leader flat pairs overlap with the
  // 4-byte grouped membership copy. After routing, only the grouped copy is
  // retained. The root fanout is the largest fanout in this configuration.
  const uint64_t root_flat = n * root_fanout * 8;
  const uint64_t root_grouped = n * root_fanout * 4;
  // MergeBuckets copies pq.top() into remaining_clusters. Freed Parlay pages
  // can remain resident until a later trim, so bound this second membership
  // generation as part of the root routing phase.
  plan.root_merge_copy_bytes = root_grouped;
  const uint64_t root_ids = n * sizeof(uint32_t);
  const uint64_t leader_state = max_leaders * dim * sizeof(float) * 2;
  // root_flat is released before the grouped membership plus MergeBuckets
  // copy becomes resident.  Charge the larger phase, rather than adding two
  // allocations that are never simultaneously live.
  const uint64_t root_membership_peak =
      std::max(root_flat, root_grouped + plan.root_merge_copy_bytes);
  plan.routing_bytes = plan.fixed_bytes + hash_state + root_ids +
                       root_membership_peak + plan.workspace_bytes + leader_state;
  // The bounded root subtree batch holds grouped memberships. Its transient
  // child arrays are charged to the 256 MiB or 4% process guard below.
  plan.leaf_bytes = plan.fixed_bytes + hash_state + root_ids +
                    root_grouped + plan.workspace_bytes;
  plan.allocation_bytes = plan.leaf_bytes + n + threads * 1024 * sizeof(uint64_t);
  // Serialization is reported separately and never sets construction floor.
  const uint64_t output_block = std::min<uint64_t>(n, 1000000);
  plan.serialization_bytes = output_block * 24 +
      2 * output_block * final_degree * sizeof(uint32_t) + MiB;
  plan.final_bytes = plan.fixed_bytes + n * sizeof(uint32_t) +
                     n * final_extra_per_node + threads * final_degree * 32;
  plan.base_floor_bytes = std::max({plan.routing_bytes, plan.leaf_bytes,
                                    plan.allocation_bytes, plan.final_bytes});
  plan.mandatory_floor_bytes =
      n * (plan.evidence_floor - base_capacity) * candidate_bytes;
  const uint64_t resident_floor_bytes =
      plan.base_floor_bytes + plan.mandatory_floor_bytes;
  plan.minimum_total_bytes = std::max<uint64_t>(
      resident_floor_bytes + 256 * MiB,
      (resident_floor_bytes * 25 + 23) / 24);
  plan.minimum_total_bytes =
      ((plan.minimum_total_bytes + MiB - 1) / MiB) * MiB;
  if (total) {
    plan.guard_bytes = std::max<uint64_t>(256 * MiB, plan.total_bytes / 25);
    if (plan.total_bytes < resident_floor_bytes)
      throw std::runtime_error("total target cannot hold mandatory B64 state/workspace");
    // A zero residual is valid. Never sacrifice the mandatory B64 row to
    // satisfy the guard, and never require overflow consumption.
    plan.overflow_bytes = plan.total_bytes > resident_floor_bytes + plan.guard_bytes
        ? plan.total_bytes - resident_floor_bytes - plan.guard_bytes : 0;
  } else if (debug) {
    plan.overflow_bytes = aea_parse_mib(debug, "PIPNN_AEA_OVERFLOW_MIB");
  }
  plan.overflow_bytes = (plan.overflow_bytes / (32 * candidate_bytes)) *
                        (32 * candidate_bytes);
  const uint64_t maximum_useful_slots = std::min<uint64_t>(
      n * (plan.maximum_logical_capacity - plan.evidence_floor),
      uint64_t(UINT32_MAX) * 32);
  plan.overflow_bytes = std::min(plan.overflow_bytes,
                                 maximum_useful_slots * candidate_bytes);
  if (emit) std::cout << "AEA_MEMORY_PLAN interface="
            << (plan.total_interface ? "total" : "debug_overflow")
            << " n=" << n << " dim=" << dim << " element_bytes=" << element_bytes
            << " threads=" << threads << " base_capacity=" << base_capacity
            << " record_bytes=" << candidate_bytes
            << " fixed_bytes=" << plan.fixed_bytes
            << " query_resident_bytes=0 ground_truth_resident_bytes=0"
            << " workspace_memory_bytes=" << plan.workspace_bytes
            << " prehash_transient_bytes=" << plan.prehash_transient_bytes
            << " routing_without_overflow=" << plan.routing_bytes
            << " root_merge_copy_bytes=" << plan.root_merge_copy_bytes
            << " leaf_without_overflow=" << plan.leaf_bytes
            << " allocation_without_overflow=" << plan.allocation_bytes
            << " final_without_overflow=" << plan.final_bytes
            << " serialization_bytes=" << plan.serialization_bytes
            << " base_floor_bytes=" << plan.base_floor_bytes
            << " evidence_floor=" << plan.evidence_floor
            << " mandatory_floor_bytes=" << plan.mandatory_floor_bytes
            << " mandatory_candidate_bytes=" << plan.mandatory_candidate_bytes
            << " minimum_total_bytes=" << plan.minimum_total_bytes
            << " guard_bytes=" << plan.guard_bytes
            << " total_bytes=" << plan.total_bytes
            << " adaptive_overflow_budget_bytes=" << plan.overflow_bytes
            << " maximum_logical_capacity=" << plan.maximum_logical_capacity
            << " overflow_budget_bytes=" << plan.overflow_bytes << '\n';
  return plan;
}

}  // namespace parlayANN

// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once
#include "decision_precision_diagnostic.h"
#include "mips_prune_policy.h"

#include <math.h>

#include <algorithm>
#include <queue>
#include <numeric>
#include <atomic>
#include <cstring>
#include <type_traits>
template<class Point,class=void>struct PipNNSignedCarrier:std::false_type{};
template<class Point>struct PipNNSignedCarrier<Point,std::void_t<decltype(Point::signed_byte_carrier)>>:std::bool_constant<Point::signed_byte_carrier>{};
#include <Eigen/Dense>
#include <cfloat>


#include "../utils/graph.h"
#include "../utils/hash.h"
#include "../../../parlay/include/parlay/parallel.h"
#include "../../../parlay/include/parlay/primitives.h"
#include "../../../parlay/include/parlay/random.h"

#include "hwy/contrib/sort/vqsort-inl.h"
#include "hwy/contrib/sort/vqsort.h"
#include "hwy/highway.h"
#include <absl/types/span.h>


// #ifndef NDEBUG
#define CHECK(expr)                                                        \
  do {                                                                     \
    if (!(expr)) {                                                         \
      std::cerr << "CHECK FAILED: " << #expr << "\n    File: " << __FILE__ \
                << "\n    Line: " << __LINE__ << std::endl;                \
      std::abort();                                                        \
    }                                                                      \
  } while (0)
//#else
//#define CHECK(expr) ((void)0)
//#endif

namespace parlayANN {


std::atomic<size_t> dist_traversed(0);

//using GraphH = NghDistGraphHeap<uint32_t>;

// Forward declarations (defined later in this file).
template <class Iter> void VQSort(Iter b, Iter e);
double packFloatAndInt(float float_val, uint16_t int_val);
std::pair<float, uint32_t> unpackDouble(double packed_val);

template <typename Point, typename PointRange, typename indexType>
struct cluster_utils {
  using distanceType = typename Point::distanceType;
  using edge = std::pair<indexType, indexType>;
  using pid = std::pair<indexType, distanceType>;
  using GraphI = Graph<indexType>;
  using PR = PointRange;

  // using sketchHash = SketchHash<Point,PointRange,indexType>;
  // using GraphH = LSHGraph<indexType, float, sketchHash>;


  static constexpr indexType kNullId = std::numeric_limits<indexType>::max();
  static constexpr distanceType kNullDist =
      std::numeric_limits<distanceType>::max();
  using labelled_edge = std::pair<edge, distanceType>;
  static constexpr labelled_edge kNullEdge = {{kNullId, kNullId}, kNullDist};

  // robustPrune routine as found in DiskANN paper, with the exception that the
  // new candidate set is added to the field new_nbhs instead of directly
  // replacing the out_nbh of p
  template <class NghContainer>
  static void robustPrune(const NghContainer& neighbors, indexType p, PR &Points,
                          GraphI &G, double alpha, size_t degree) {
    // add out neighbors of p to the candidate set.
    thread_local std::vector<pid> candidates;
    candidates.clear();
    for (auto nghdist : neighbors) {
      candidates.push_back(std::make_pair(nghdist.point_id, nghdist.norm));
    }
    // Sort the candidate set by distance from p using VQSort.
    // Pack (distance, local_index) into doubles, sort, then unpack.
    thread_local std::vector<double> packed;
    packed.resize(candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i) {
      packed[i] = packFloatAndInt(candidates[i].second, static_cast<uint16_t>(i));
    }
    VQSort(packed.data(), packed.data() + packed.size());
    thread_local std::vector<pid> sorted_candidates;
    sorted_candidates.resize(candidates.size());
    for (size_t i = 0; i < packed.size(); ++i) {
      auto [dist, local_idx] = unpackDouble(packed[i]);
      sorted_candidates[i] = candidates[local_idx];
    }
    std::swap(candidates, sorted_candidates);

    // No need to remove duplicates, as the neighbors container is
    // constructed out of a hash_set.

    std::vector<int> new_nbhs;
    new_nbhs.reserve(degree);

    float inverse_alpha = 1.0 / alpha;
    size_t candidate_idx = 0;
    while (new_nbhs.size() < degree && candidate_idx < candidates.size()) {
      // Don't need to do modifications.
      const indexType p_prime = candidates[candidate_idx].first;
      const auto dist_pprime = candidates[candidate_idx].second * inverse_alpha;
      candidate_idx++;
      if (p_prime == p) continue;
      bool add = true;
      for (auto p_star : new_nbhs) {
        const auto dist_starprime = Points[p_star].distance(Points[p_prime]);
        if (dist_starprime <= dist_pprime) {
          add = false;
          break;
        }
      }
      if (add) new_nbhs.push_back(p_prime);
    }
    dist_traversed.fetch_add(candidate_idx, std::memory_order_relaxed);
    G[p].update_neighbors(new_nbhs);
  }

  // Two-Witness prune keeps a candidate until two independently selected
  // neighbors certify it as locally replaceable. Candidate ordering and the
  // alpha test remain identical to released PiPNN RobustPrune.
  template <class NghContainer>
  static void robustPruneTwoWitness(const NghContainer& neighbors, indexType p,
                                    PR &Points, GraphI &G, double alpha,
                                    size_t degree,
                                    distanceType distance_shift = 0) {
    thread_local std::vector<pid> candidates;
    candidates.clear();
    for (auto nghdist : neighbors)
      candidates.emplace_back(nghdist.point_id, nghdist.norm);
    thread_local std::vector<double> packed;
    packed.resize(candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i)
      packed[i] = packFloatAndInt(candidates[i].second,
                                  static_cast<uint16_t>(i));
    VQSort(packed.data(), packed.data() + packed.size());
    thread_local std::vector<pid> sorted_candidates;
    sorted_candidates.resize(candidates.size());
    for (size_t i = 0; i < packed.size(); ++i) {
      auto [dist, local_idx] = unpackDouble(packed[i]);
      sorted_candidates[i] = candidates[local_idx];
    }
    std::swap(candidates, sorted_candidates);

    std::vector<int> new_nbhs;
    new_nbhs.reserve(degree);
    const float inverse_alpha = 1.0 / alpha;
    size_t candidate_idx = 0;
    while (new_nbhs.size() < degree && candidate_idx < candidates.size()) {
      const indexType candidate = candidates[candidate_idx].first;
      const auto threshold = std::max<distanceType>(
          0, candidates[candidate_idx].second + distance_shift) * inverse_alpha;
      ++candidate_idx;
      if (candidate == p) continue;
      unsigned witnesses = 0;
      for (auto selected : new_nbhs) {
        const auto witness_distance = std::max<distanceType>(
            0, Points[selected].distance(Points[candidate]) + distance_shift);
        if (witness_distance <= threshold &&
            ++witnesses == 2)
          break;
      }
      if (witnesses < 2) new_nbhs.push_back(candidate);
    }
    dist_traversed.fetch_add(candidate_idx, std::memory_order_relaxed);
    G[p].update_neighbors(new_nbhs);
  }

  // Selection-only form used by exact in-place candidate-to-final handoff.
  // It is intentionally the same algorithm and ordering as the writer above.
  template <class NghContainer>
  static void robustPruneTwoWitnessIds(const NghContainer& neighbors,
                                       indexType p, PR& Points, double alpha,
                                       size_t degree,
                                       std::vector<indexType>& new_nbhs,
                                       distanceType distance_shift = 0) {
    thread_local std::vector<pid> candidates;
    candidates.clear();
    for (auto nghdist : neighbors)
      candidates.emplace_back(nghdist.point_id, nghdist.norm);
    thread_local std::vector<double> packed;
    packed.resize(candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i)
      packed[i] = packFloatAndInt(candidates[i].second,
                                  static_cast<uint16_t>(i));
    VQSort(packed.data(), packed.data() + packed.size());
    thread_local std::vector<pid> sorted_candidates;
    sorted_candidates.resize(candidates.size());
    for (size_t i = 0; i < packed.size(); ++i) {
      auto [dist, local_idx] = unpackDouble(packed[i]);
      sorted_candidates[i] = candidates[local_idx];
    }
    std::swap(candidates, sorted_candidates);
    for(size_t q=1;q<candidates.size();++q)
      PIPNN_PRECISION_RECORD(7,p,p,candidates[q].first,candidates[q-1].first,candidates[q].second,candidates[q-1].second,candidates[q].second > candidates[q-1].second);
    new_nbhs.clear();
    new_nbhs.reserve(degree);
    const float inverse_alpha = 1.0 / alpha;
    size_t candidate_idx = 0;
    while (new_nbhs.size() < degree && candidate_idx < candidates.size()) {
      const indexType candidate = candidates[candidate_idx].first;
      const auto threshold = std::max<distanceType>(
          0, candidates[candidate_idx].second + distance_shift) * inverse_alpha;
      ++candidate_idx;
      if (candidate == p) continue;
      unsigned witnesses = 0;
      for (auto selected : new_nbhs) {
        const auto witness_distance = std::max<distanceType>(
            0, Points[selected].distance(Points[candidate]) + distance_shift);
        if (witness_distance <= threshold &&
            ++witnesses == 2) break;
      }
      if (witnesses < 2) new_nbhs.push_back(candidate);
    }
    dist_traversed.fetch_add(candidate_idx, std::memory_order_relaxed);
  }

  // Selection-only form of released one-witness RobustPrune.  This exists for
  // the isolated 1W control target so it can share TW's in-place layout without
  // accidentally executing Two-Witness semantics.
  template <class NghContainer>
  static void robustPruneIds(const NghContainer& neighbors, indexType p,
                             PR& Points, double alpha, size_t degree,
                             std::vector<indexType>& new_nbhs) {
    thread_local std::vector<pid> candidates;
    candidates.clear();
    for (auto nghdist : neighbors)
      candidates.emplace_back(nghdist.point_id, nghdist.norm);
    thread_local std::vector<double> packed;
    packed.resize(candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i)
      packed[i] = packFloatAndInt(candidates[i].second,
                                  static_cast<uint16_t>(i));
    VQSort(packed.data(), packed.data() + packed.size());
    thread_local std::vector<pid> sorted_candidates;
    sorted_candidates.resize(candidates.size());
    for (size_t i = 0; i < packed.size(); ++i) {
      auto [dist, local_idx] = unpackDouble(packed[i]);
      sorted_candidates[i] = candidates[local_idx];
    }
    std::swap(candidates, sorted_candidates);
    new_nbhs.clear();
    new_nbhs.reserve(degree);
    const float inverse_alpha = 1.0 / alpha;
    size_t candidate_idx = 0;
    while (new_nbhs.size() < degree && candidate_idx < candidates.size()) {
      const indexType candidate = candidates[candidate_idx].first;
      const auto threshold = candidates[candidate_idx].second * inverse_alpha;
      ++candidate_idx;
      if (candidate == p) continue;
      bool occluded = false;
      for (auto selected : new_nbhs) {
        const auto precision_exact = Points[selected].distance(Points[candidate]);
        PIPNN_PRECISION_RECORD(4, p, selected, candidate, p, precision_exact, threshold, precision_exact <= threshold);
        if (precision_exact <= threshold) {
          occluded = true;
          break;
        }
      }
      PIPNN_PRECISION_RECORD(5, p, p, candidate, 0, candidates[candidate_idx-1].second, threshold, !occluded);
      if (!occluded) new_nbhs.push_back(candidate);
    }
    dist_traversed.fetch_add(candidate_idx, std::memory_order_relaxed);
  }

  // Non-metric MIPS fallback for the isolated TW target.  A lune/RNG
  // occlusion test is undefined for -dot scores, so retain the nearest WARD
  // evidence directly (up to the physical final-degree budget).
  template <class NghContainer>
  static void nearestIds(const NghContainer& neighbors, indexType p,
                         size_t degree, std::vector<indexType>& new_nbhs) {
    thread_local std::vector<pid> candidates;
    candidates.clear();
    for (auto nghdist : neighbors)
      if (nghdist.point_id != p)
        candidates.emplace_back(nghdist.point_id, nghdist.norm);
    std::sort(candidates.begin(), candidates.end(),
              [](const pid& a, const pid& b) {
                if (a.second != b.second) return a.second < b.second;
                return a.first < b.first;
              });
    new_nbhs.clear();
    const size_t keep = std::min(degree, candidates.size());
    new_nbhs.reserve(degree);
    for (size_t i = 0; i < keep; ++i) new_nbhs.push_back(candidates[i].first);
    dist_traversed.fetch_add(keep, std::memory_order_relaxed);
  }

  // Explicit non-metric dispatch; no alpha parameter. The caller supplies the
  // existing shared norm table when available, avoiding full-N worker arrays.
  template<class NghContainer>
  static void robustPruneMIPS(const NghContainer& neighbors,indexType p,
      PR& Points,size_t degree,std::vector<indexType>& out,
      const std::vector<float>* norms=nullptr) {
    // Restores the existing angular/norm two-witness predicate, with native
    // negative-dot decoding and explicit zero-norm handling. The same policy
    // is shared by initialization, early repruning and reverse overflow.
    thread_local std::vector<pid> candidates;
    candidates.clear();
    for(auto c:neighbors)if(c.point_id!=p){
      candidates.emplace_back(c.point_id,c.norm);
    }
    std::sort(candidates.begin(),candidates.end(),[](const pid& a,const pid& b){
      return a.second<b.second||(a.second==b.second&&a.first<b.first);});
    out.clear();out.reserve(degree);
    for(auto c:candidates){
      if(out.size()==degree)break;
      if(!mips_prune::occluded(p,c.first,c.second,out.size(),
          [&](size_t j){return out[j];},
          [&](indexType a,indexType b){return Points[a].distance(Points[b]);},
          [&](indexType id)->double {
            if(norms)return (*norms)[id];
            double sum=0;auto point=Points[id];
            for(long k=0;k<Points.dimension();++k)sum+=double(point[k])*point[k];
            return std::sqrt(sum);
          },mips_prune::strategy()))out.push_back(c.first);
    }
  }

  // Research-only exact locally calibrated Two-Witness oracle. Candidate
  // relevance/order uses the configured retrieval dissimilarity.  Only the
  // cross-anchor redundancy predicate is replaced by reciprocal empirical
  // percentiles computed from the complete per-anchor evidence distributions.
  // `ecdf_offsets` and `ecdf_values` contain sorted native dissimilarities.
  template <class NghContainer>
  static void robustPruneReciprocalRankIds(
      const NghContainer& neighbors, indexType p, PR& Points, size_t degree,
      const std::vector<uint64_t>& ecdf_offsets,
      const std::vector<distanceType>& ecdf_values,
      std::vector<indexType>& new_nbhs) {
    uint64_t local_witness_tests = 0, local_pair_distances = 0;
    uint64_t local_source_gate_tests = 0, local_source_gate_rejects = 0;
    uint64_t local_source_gate_lt2_candidates = 0;
    uint64_t local_source_gate_eligible = 0;
    uint64_t local_interval_true = 0, local_interval_false = 0;
    uint64_t local_interval_fallback = 0;
    thread_local std::vector<pid> candidates;
    candidates.clear();
    for (auto ngh : neighbors) {
      if (ngh.point_id == p) continue;
      const indexType v = ngh.point_id;
      candidates.emplace_back(v, Points[p].distance(Points[v]));
    }
    std::sort(candidates.begin(), candidates.end(), [](const pid& a,
                                                        const pid& b) {
      if (a.second != b.second) return a.second < b.second;
      return a.first < b.first;
    });

    struct rank_interval { double lo, hi, exact; bool is_exact; uint64_t bin; };
    auto rank_of = [&](indexType anchor, indexType other, distanceType value,
                       bool local_known) -> rank_interval {
      const uint64_t begin = ecdf_offsets[anchor];
      const uint64_t end = ecdf_offsets[anchor + 1];
      const auto first = ecdf_values.begin() + begin;
      const auto last = ecdf_values.begin() + end;
      const uint64_t le = static_cast<uint64_t>(
          std::upper_bound(first, last, value) - first);
      const double exact = static_cast<double>(1 + le) /
                           static_cast<double>(1 + end - begin);
      return {exact, exact, exact, true, 0};
    };
    struct delta_interval { double lo, hi, exact; uint64_t remat; };
    auto reciprocal = [&](indexType a, indexType b, distanceType value,
                          bool a_known, bool b_known) -> delta_interval {
      const auto ab = rank_of(a, b, value, a_known);
      const auto ba = rank_of(b, a, value, b_known);
      return {std::max(ab.lo, ba.lo), std::max(ab.hi, ba.hi),
              std::max(ab.exact, ba.exact), ab.bin + ba.bin};
    };
    thread_local std::vector<delta_interval> selected_source_deltas;
    thread_local std::vector<size_t> eligible_selected;
    selected_source_deltas.clear();
    eligible_selected.clear();
    selected_source_deltas.reserve(degree);

    new_nbhs.clear();
    new_nbhs.reserve(degree);
    size_t candidate_idx = 0;
    while (new_nbhs.size() < degree && candidate_idx < candidates.size()) {
      const indexType candidate = candidates[candidate_idx++].first;
      const distanceType source_value = candidates[candidate_idx - 1].second;
      const auto source_candidate = reciprocal(p, candidate, source_value,
                                               true, false);
      // Stage 1 is source-local and therefore needs no selected-candidate
      // distance.  It is a necessary condition for being an RR2W witness.
      // Build the complete eligible set before performing any cross-anchor
      // tests.  Ordering eligible witnesses by their calibrated source
      // strength changes only evaluation order, not the existential 2W
      // decision.
      eligible_selected.clear();
      for (size_t selected_idx = 0; selected_idx < new_nbhs.size();
           ++selected_idx) {
        ++local_source_gate_tests;
        const auto& source_selected = selected_source_deltas[selected_idx];
        if (source_selected.exact < source_candidate.exact)
          eligible_selected.push_back(selected_idx);
        else
          ++local_source_gate_rejects;
      }
      local_source_gate_eligible += eligible_selected.size();
      if (eligible_selected.size() < 2) {
        ++local_source_gate_lt2_candidates;
        new_nbhs.push_back(candidate);
        selected_source_deltas.push_back(source_candidate);
        continue;
      }
      std::sort(eligible_selected.begin(), eligible_selected.end(),
                [&](size_t a, size_t b) {
                  const auto& da = selected_source_deltas[a];
                  const auto& db = selected_source_deltas[b];
                  if (da.exact != db.exact) return da.exact < db.exact;
                  return new_nbhs[a] < new_nbhs[b];
                });
      unsigned witnesses = 0;
      for (size_t selected_idx : eligible_selected) {
        const indexType selected = new_nbhs[selected_idx];
        ++local_witness_tests;
        const auto& source_selected = selected_source_deltas[selected_idx];
        const distanceType cross_value = Points[selected].distance(Points[candidate]);
        ++local_pair_distances;
        const auto cross = reciprocal(selected, candidate, cross_value,
                                      false, false);
        bool is_witness;
        if (source_selected.hi < source_candidate.lo &&
            cross.hi < source_candidate.lo) {
          ++local_interval_true;
          is_witness = true;
        } else if (source_selected.lo >= source_candidate.hi ||
                   cross.lo >= source_candidate.hi) {
          ++local_interval_false;
          is_witness = false;
        } else {
          ++local_interval_fallback;
          is_witness = source_selected.exact < source_candidate.exact &&
                       cross.exact < source_candidate.exact;
        }
        if (is_witness && ++witnesses == 2)
          break;
      }
      if (witnesses < 2) {
        new_nbhs.push_back(candidate);
        selected_source_deltas.push_back(source_candidate);
      }
    }
    dist_traversed.fetch_add(candidate_idx, std::memory_order_relaxed);
  }

  // Research-only similarity-oblivious finalizer. Fine residual cones are
  // coarsened through their 12-bit prefix trie until at most `degree` active
  // cells remain. Each active cell contributes its best source-dissimilarity
  // representative. No candidate-candidate geometry is consulted.
  template <class NghContainer>
  static void coneTrieIds(const NghContainer& neighbors, indexType p,
                          PR& Points, size_t degree,
                          std::vector<indexType>& new_nbhs) {
    struct Candidate { indexType id; float score; uint16_t hash; };
    struct Cell {
      uint8_t depth = 0;
      std::vector<size_t> members;
      bool active = true;
    };
    thread_local std::vector<Candidate> candidates;
    candidates.clear();
    for (auto ngh : neighbors) {
      if (ngh.point_id == p) continue;
      candidates.push_back({ngh.point_id,
          static_cast<float>(Points[p].distance(Points[ngh.point_id])),
          static_cast<uint16_t>(ngh.hash)});
    }
    new_nbhs.clear();
    if (candidates.empty() || degree == 0) return;

    std::vector<Cell> cells;
    cells.push_back(Cell{});
    cells[0].members.resize(candidates.size());
    std::iota(cells[0].members.begin(), cells[0].members.end(), 0);
    auto cmp = [&](size_t a, size_t b) {
      if (cells[a].members.size() != cells[b].members.size())
        return cells[a].members.size() < cells[b].members.size();
      return cells[a].depth > cells[b].depth;
    };
    std::priority_queue<size_t, std::vector<size_t>, decltype(cmp)> pending(cmp);
    pending.push(0);
    size_t active = 1;
    while (active < degree && !pending.empty()) {
      const size_t ci = pending.top(); pending.pop();
      if (!cells[ci].active || cells[ci].depth == 12 ||
          cells[ci].members.size() <= 1) continue;
      Cell left, right;
      left.depth = right.depth = cells[ci].depth + 1;
      const unsigned shift = 11 - cells[ci].depth;
      for (size_t x : cells[ci].members) {
        ((candidates[x].hash >> shift) & 1 ? right.members : left.members)
            .push_back(x);
      }
      cells[ci].active = false;
      const size_t li = cells.size();
      if (!left.members.empty()) cells.push_back(std::move(left));
      const size_t ri = cells.size();
      if (!right.members.empty()) cells.push_back(std::move(right));
      const size_t children = cells.size() - li;
      if (children == 0) break;
      active += children - 1;
      for (size_t x = li; x < cells.size(); ++x) pending.push(x);
      (void)ri;
    }

    new_nbhs.reserve(std::min(degree, active));
    for (const Cell& cell : cells) {
      if (!cell.active || cell.members.empty()) continue;
      size_t best = cell.members[0];
      for (size_t x : cell.members)
        if (candidates[x].score < candidates[best].score ||
            (candidates[x].score == candidates[best].score &&
             candidates[x].id < candidates[best].id)) best = x;
      new_nbhs.push_back(candidates[best].id);
    }
    if (new_nbhs.size() > degree) new_nbhs.resize(degree);
    dist_traversed.fetch_add(candidates.size(), std::memory_order_relaxed);
  }

  // Research-only MIPS control using only the configured dissimilarity.
  // Antichain mode counts two occluders only when neither occludes the other
  // relative to the same source. This tests whether two witnesses really
  // denote two independent routes, without inspecting norms or coordinates.
  template <class NghContainer>
  static void robustPruneMipsAntichainIds(
      const NghContainer& neighbors, indexType p, PR& Points, double alpha,
      size_t degree, std::vector<indexType>& new_nbhs, bool require_antichain) {
    thread_local std::vector<pid> candidates;
    candidates.clear();
    for (auto nghdist : neighbors) {
      if (nghdist.point_id == p) continue;
      const indexType v = nghdist.point_id;
      candidates.emplace_back(v, Points[p].distance(Points[v]));
    }
    std::sort(candidates.begin(), candidates.end(), [](const pid& a,
                                                        const pid& b) {
      if (a.second != b.second) return a.second < b.second;
      return a.first < b.first;
    });

    new_nbhs.clear();
    new_nbhs.reserve(degree);
    const double inverse_alpha = 1.0 / alpha;
    size_t candidate_idx = 0;
    thread_local std::vector<indexType> occluders;
    while (new_nbhs.size() < degree && candidate_idx < candidates.size()) {
      const indexType candidate = candidates[candidate_idx].first;
      const double threshold = candidates[candidate_idx].second * inverse_alpha;
      ++candidate_idx;
      occluders.clear();
      bool rejected = false;
      for (const indexType selected : new_nbhs) {
        if (Points[selected].distance(Points[candidate]) > threshold) continue;
        if (!require_antichain) {
          if (!occluders.empty()) { rejected = true; break; }
          occluders.push_back(selected);
          continue;
        }
        for (const indexType prior : occluders) {
          const bool selected_occludes_prior =
              Points[selected].distance(Points[prior]) <=
              Points[p].distance(Points[prior]) * inverse_alpha;
          const bool prior_occludes_selected =
              Points[prior].distance(Points[selected]) <=
              Points[p].distance(Points[selected]) * inverse_alpha;
          if (!selected_occludes_prior && !prior_occludes_selected) {
            rejected = true;
            break;
          }
        }
        if (rejected) break;
        occluders.push_back(selected);
      }
      if (!rejected) new_nbhs.push_back(candidate);
    }
    dist_traversed.fetch_add(candidate_idx, std::memory_order_relaxed);
  }

  // Research-only dual-geometry finalization for MIPS.  Candidate relevance
  // and ordering remain the native MIPS score stored by WARD, while the
  // witness relation is evaluated in a separate, non-negative geometry.
  // This deliberately does not alter the stable MIPS fallback above.
  template <class NghContainer>
  static void robustPruneMipsDualGeometryIds(
      const NghContainer& neighbors, indexType p, PR& Points, double alpha,
      size_t degree, std::vector<indexType>& new_nbhs,
      bool angular, bool norm_guard,
      const std::vector<float>* point_norms = nullptr) {
    auto norm_of=[&](indexType id)->double {
      if(point_norms)return (*point_norms)[id];
      auto point=Points[id];double sum=0;
      for(long k=0;k<Points.dimension();++k)sum+=double(point[k])*point[k];
      return std::sqrt(sum);
    };
    thread_local std::vector<pid> candidates;
    candidates.clear();
    for (auto nghdist : neighbors)
      if (nghdist.point_id != p)
        candidates.emplace_back(nghdist.point_id, nghdist.norm);
    {
      // Both acquisition and selected two-hop rescoring store native -dot.
      // MIPS relevance remains descending dot, NOT cosine or recovered L2.
      std::sort(candidates.begin(), candidates.end(),
                [](const pid& a, const pid& b) {
                  if (a.second != b.second) return a.second < b.second;
                  return a.first < b.first;
                });
    }

    auto witness_distance = [&](indexType x, indexType y) {
      auto px = Points[x];
      auto py = Points[y];
      double dot = 0.0, nx = 0.0, ny = 0.0, l2 = 0.0;
      for (long j = 0; j < Points.dimension(); ++j) {
        const double xv = static_cast<double>(px[j]);
        const double yv = static_cast<double>(py[j]);
        if (angular) {
          dot += xv * yv;
          nx += xv * xv;
          ny += yv * yv;
        } else {
          const double delta = xv - yv;
          l2 += delta * delta;
        }
      }
      if (!angular) return l2;
      if (nx == 0.0 || ny == 0.0) return 3.14159265358979323846;
      const double cosine = std::max(-1.0, std::min(1.0, dot / std::sqrt(nx * ny)));
      return std::acos(cosine);
    };

    new_nbhs.clear();
    new_nbhs.reserve(degree);
    thread_local std::vector<double> selected_source_cosines;
    selected_source_cosines.clear();
    selected_source_cosines.reserve(degree);
    const double inverse_alpha = 1.0 / alpha;
    size_t candidate_idx = 0;
    while (new_nbhs.size() < degree && candidate_idx < candidates.size()) {
      const pid candidate_entry = candidates[candidate_idx++];
      const indexType candidate = candidate_entry.first;
      if (angular) {
        const double source_norm = norm_of(p);
        const double candidate_norm = norm_of(candidate);
        const double candidate_dot = -static_cast<double>(candidate_entry.second);
        const double candidate_cosine =
            (source_norm > 0.0 && candidate_norm > 0.0)
                ? candidate_dot / (source_norm * candidate_norm)
                : -1.0;
        unsigned witnesses = 0;
        auto pv = Points[candidate];
        for (size_t selected_idx = 0; selected_idx < new_nbhs.size();
             ++selected_idx) {
          const indexType selected = new_nbhs[selected_idx];
          const double selected_norm=norm_of(selected);
          if (norm_guard && selected_norm < candidate_norm)
            continue;

          if (!(selected_source_cosines[selected_idx] > candidate_cosine))
            continue;

          auto pw = Points[selected];
          double witness_dot = 0.0;
          for (long j = 0; j < Points.dimension(); ++j)
            witness_dot += static_cast<double>(pw[j]) *
                           static_cast<double>(pv[j]);
          const double denominator=selected_norm*candidate_norm;
          if(!(denominator>0))continue;
          const double witness_cosine = witness_dot / denominator;
          if (witness_cosine > candidate_cosine && ++witnesses == 2) break;
        }
        if (witnesses < 2) {
          new_nbhs.push_back(candidate);
          selected_source_cosines.push_back(candidate_cosine);
        }
        continue;
      }
      const double threshold = witness_distance(p, candidate) * inverse_alpha;
      unsigned witnesses = 0;
      for (auto selected : new_nbhs) {
        if (witness_distance(selected, candidate) <= threshold &&
            ++witnesses == 2)
          break;
      }
      if (witnesses < 2) new_nbhs.push_back(candidate);
    }
    dist_traversed.fetch_add(candidate_idx, std::memory_order_relaxed);
  }

  // Diagnostic only: approximate each sampled candidate's distance to the
  // convex hull of the other candidates.  Translation by the source cancels,
  // so the calculation can use base vectors directly.  Frank-Wolfe is run on
  // a precomputed Gram matrix; no graph state is modified.
  template <class GraphH>
  static void reportMipsConvexSupportResidual(GraphH& graph, PR& Points) {
    constexpr size_t sample_count = 128;
    constexpr size_t fw_iterations = 128;
    const size_t actual_samples = std::min(sample_count, Points.size());
    std::vector<std::vector<double>> sample_residuals(actual_samples);
    parlay::parallel_for(0, actual_samples, [&](size_t sample) {
      const size_t source = (sample * Points.size()) / actual_samples;
      std::vector<indexType> ids;
      for (auto candidate : graph.Neighbors(source))
        if (candidate.point_id != source) ids.push_back(candidate.point_id);
      const size_t m = ids.size();
      if (m < 2) return;
      std::vector<double> gram(m * m, 0.0);
      for (size_t i = 0; i < m; ++i) {
        auto pi = Points[ids[i]];
        for (size_t j = 0; j <= i; ++j) {
          auto pj = Points[ids[j]];
          double dot = 0.0;
          for (long k = 0; k < Points.dimension(); ++k)
            dot += static_cast<double>(pi[k]) * static_cast<double>(pj[k]);
          gram[i * m + j] = gram[j * m + i] = dot;
        }
      }
      auto ps = Points[source];
      double source_norm = 0.0;
      std::vector<double> source_dot(m, 0.0);
      for (long k = 0; k < Points.dimension(); ++k) {
        const double value = static_cast<double>(ps[k]);
        source_norm += value * value;
      }
      for (size_t i = 0; i < m; ++i) {
        auto pi = Points[ids[i]];
        for (long k = 0; k < Points.dimension(); ++k)
          source_dot[i] += static_cast<double>(ps[k]) *
                           static_cast<double>(pi[k]);
      }
      auto& output = sample_residuals[sample];
      output.reserve(m);
      for (size_t target = 0; target < m; ++target) {
        size_t atom = target == 0 ? 1 : 0;
        double best = std::numeric_limits<double>::infinity();
        for (size_t j = 0; j < m; ++j) if (j != target) {
          const double d2 = gram[j*m+j] + gram[target*m+target]
                            - 2.0 * gram[j*m+target];
          if (d2 < best) { best = d2; atom = j; }
        }
        std::vector<double> lambda(m, 0.0), gy(m, 0.0);
        lambda[atom] = 1.0;
        for (size_t j = 0; j < m; ++j) gy[j] = gram[j*m+atom];
        double yy = gram[atom*m+atom];
        double yt = gram[atom*m+target];
        for (size_t iteration = 0; iteration < fw_iterations; ++iteration) {
          size_t next = target == 0 ? 1 : 0;
          double min_gradient = std::numeric_limits<double>::infinity();
          for (size_t j = 0; j < m; ++j) if (j != target) {
            const double gradient = gy[j] - gram[j*m+target];
            if (gradient < min_gradient) { min_gradient = gradient; next = j; }
          }
          const double numerator = yy - yt - gy[next] + gram[next*m+target];
          const double denominator = yy - 2.0*gy[next] + gram[next*m+next];
          const double step = denominator > 0.0
              ? std::max(0.0, std::min(1.0, numerator / denominator)) : 0.0;
          if (step <= 1e-12) break;
          const double old_yy = yy;
          const double old_y_next = gy[next];
          yy = (1.0-step)*(1.0-step)*old_yy
               + 2.0*step*(1.0-step)*old_y_next
               + step*step*gram[next*m+next];
          yt = (1.0-step)*yt + step*gram[next*m+target];
          for (size_t j = 0; j < m; ++j)
            gy[j] = (1.0-step)*gy[j] + step*gram[j*m+next];
        }
        const double residual2 = std::max(0.0,
            yy + gram[target*m+target] - 2.0*yt);
        const double displacement2 = std::max(1e-30,
            gram[target*m+target] + source_norm - 2.0*source_dot[target]);
        output.push_back(std::sqrt(residual2 / displacement2));
      }
    });
    std::vector<double> residuals;
    for (auto& row : sample_residuals)
      residuals.insert(residuals.end(), row.begin(), row.end());
    std::sort(residuals.begin(), residuals.end());
    auto quantile = [&](double q) {
      if (residuals.empty()) return 0.0;
      return residuals[static_cast<size_t>(q * (residuals.size()-1))];
    };
    auto fraction_below = [&](double threshold) {
      return residuals.empty() ? 0.0 :
        static_cast<double>(std::upper_bound(residuals.begin(), residuals.end(),
                                             threshold) - residuals.begin()) /
        residuals.size();
    };
    std::cout << "mips_convex_support_residual samples=" << actual_samples
              << " candidates=" << residuals.size()
              << " p10=" << quantile(0.10) << " p50=" << quantile(0.50)
              << " p90=" << quantile(0.90) << " p99=" << quantile(0.99)
              << " frac_le_1e-3=" << fraction_below(1e-3)
              << " frac_le_1e-2=" << fraction_below(1e-2)
              << " frac_le_5e-2=" << fraction_below(5e-2) << std::endl;
  }

};


template <class Iter>
void VQSort(Iter b, Iter e) {
#if HWY_MAJOR >= 1
  hwy::VQSort(b, e-b, hwy::SortAscending());
#else
  std::sort(b, e);
#endif
}

template <class Iter>
void VQPartialSort(Iter b, Iter e, size_t k) {
#if HWY_MAJOR >= 1
  hwy::VQPartialSort(b, e-b, k, hwy::SortAscending());
#else
  std::partial_sort(b, b + std::min<size_t>(k, e - b), e);
#endif
}

template <class Iter>
void VQSelect(Iter b, Iter e, size_t k) {
#if HWY_MAJOR >= 1
  hwy::VQSelect(b, e-b, k, hwy::SortAscending());
#else
  if (k < static_cast<size_t>(e - b)) std::nth_element(b, b + k, e);
#endif
}

// A union allows us to access the same piece of memory in different ways.
// Here, we can write a 'double' and read its raw 64 bits as a 'uint64_t'.
union DoubleConverter {
    double d;
    uint64_t i;
};

/**
 * @brief Packs a float and an 11-bit integer into a double.
 *
 * The function converts the float to a double, then overwrites the
 * lowest 11 bits of the double's mantissa with the provided integer value.
 *
 * @param float_val The floating-point value to pack.
 * @param int_val The 11-bit integer value to pack (must be in range [0, 2047]).
 * @return A double containing the packed data.
 */
double packFloatAndInt(float float_val, uint16_t int_val) {
    DoubleConverter converter;

    // 1. Start by converting the float to a double. This sets the sign,
    //    exponent, and the most significant bits of the mantissa correctly.
    converter.d = static_cast<double>(float_val);

    // 2. Define masks for bitwise operations.
    // Mask to clear the lowest 11 bits (all 1s except for the last 11 bits).
    // ~0ULL is all 1s. (1ULL << 11) is 2048. 2048 - 1 is 2047 (0x7FF), which is 11 set bits.
    // The inverse gives us a mask to clear those bits.
    const uint64_t clear_mask = ~((1ULL << 11) - 1);

    // Mask to ensure the input integer is only 11 bits. 0x7FF is 0...011111111111.
    const uint64_t int_mask = 0x7FF;

    // 3. Clear the lowest 11 bits of the double's integer representation.
    converter.i &= clear_mask;

    // 4. Combine (OR) the cleared integer with the 11-bit value.
    // We mask int_val to be safe and ensure no higher bits are set.
    converter.i |= (static_cast<uint64_t>(int_val) & int_mask);

    // 5. Return the result as a double.
    return converter.d;
}

/**
 * @brief Unpacks a double back into a float and an 11-bit integer.
 *
 * @param packed_val The double containing the packed data.
 * @return A std::pair containing the extracted float and the 11-bit integer.
 */
std::pair<float, uint32_t> unpackDouble(double packed_val) {
    DoubleConverter converter;
    converter.d = packed_val;

    // 1. Define the mask to extract the lowest 11 bits.
    const uint64_t extract_mask = (1ULL << 11) - 1; // 0x7FF

    // 2. Extract the 11-bit integer by ANDing with the mask.
    uint32_t extracted_int = static_cast<uint16_t>(converter.i & extract_mask);

    // 3. To get the original float value, we must clear the packed integer bits
    //    from the double's representation.
    converter.i &= ~extract_mask;

    // 4. Convert the modified double back to a float.
    float extracted_float = static_cast<float>(converter.d);

    return {extracted_float, extracted_int};
}



thread_local uint8_t* tl_dist_mat = nullptr;
thread_local uint8_t* tl_point_mat = nullptr;
size_t kMaxClusterSize = 2048;
size_t kMaxDimension = 2048;

void* get_point_mat() {
  if (tl_point_mat == nullptr) {
    // Enough space for float points.
    tl_point_mat = (uint8_t*)_mm_malloc(kMaxClusterSize * kMaxDimension * sizeof(uint32_t), 64);
    for (size_t i=0; i < kMaxClusterSize * kMaxDimension * sizeof(uint32_t); ++i) {
      tl_point_mat[i] = 0;
    }
  }
  return tl_point_mat;
}

// n by n output matrix.
void* get_dist_mat() {
  if (tl_dist_mat == nullptr) {
    // Enough space for float distances.
    tl_dist_mat = (uint8_t*)_mm_malloc(kMaxClusterSize * kMaxClusterSize * sizeof(uint32_t), 64);
    for (size_t i=0; i < kMaxClusterSize * kMaxClusterSize * sizeof(uint32_t); ++i) {
      tl_dist_mat[i] = 0;
    }
  }
  return tl_dist_mat;
}


// Materializes the output floating-point distmat into the
// thread-local dist_mat variable above.
template <typename PR, typename Seq>
float* eigen_build_distmat(Seq& seq, PR& all_points) {
  size_t n = seq.size();
  using coord_type = typename PR::Point::T;

  size_t dimension = all_points.get_dims();
  float* dist_mat = pipnn_clean_leaf_distances(n * n);
  float* point_mat = pipnn_clean_leaf_points(n * dimension);

  // Copy the points into point_mat, and convert to floats.
  for (size_t i=0; i < n; ++i) {
    coord_type* loc = (coord_type*)all_points.location(seq[i]);
    for (size_t j=0; j < dimension; ++j) {
      point_mat[i*dimension + j] = static_cast<float>(loc[j]);
    }
  }

  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> P(point_mat, n, dimension);
  // TODO: can we just do this once, offline for every point and
  // re-use?
  Eigen::Matrix<float, Eigen::Dynamic, 1> P_sq_norms = P.rowwise().squaredNorm();
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> D(dist_mat, n, n);

  // Sets the lower-triangular D = P x P^T
  D.setZero();
  D.selfadjointView<Eigen::Lower>().rankUpdate(P);

  using Point = typename PR::Point;
  // Preserve the released PiPNN acquisition semantics.  Metric points use
  // squared L2; non-metric MIPS points retain the GEMM-emitted -dot score.
  // HEC is an exact keyed reduction over whichever score the leaf generator
  // emits and must not silently change that score geometry.
  if (Point::is_metric()) {
    // Finish up square norm computation, if needed.
    for (size_t i=0; i < n; ++i) {
      for (size_t j=0; j <= i; ++j) {
        float tmp = D(i, j);
        D(i, j) = -2*tmp;
        D(i, j) += P_sq_norms[i] + P_sq_norms[j];
      }
    }
  } else {
    for (size_t i=0; i < n; ++i) {
      for (size_t j=0; j <= i; ++j) {
        D(i, j) = -1*D(i, j);
      }
    }
  }
  // The caller must consume the exact workspace populated above.  In the
  // memory-clean configuration this is not the legacy get_dist_mat() buffer.
  return dist_mat;
}

thread_local uint8_t* tile_point_mat   = nullptr;
thread_local uint8_t* tile_dist_mat    = nullptr;
// Keep each worker's top-level routing distance matrix near 1 MiB at the
// released maximum of 1000 leaders.  This changes blocking only, never the
// set of distances, leaders, or assignments.
constexpr size_t kMaxTileSize   = 256;
constexpr size_t kMaxLeaderSize = 2048;

static void* get_tile_point_mat() {
  if (!tile_point_mat)
    tile_point_mat = (uint8_t*)_mm_malloc(
      kMaxTileSize * kMaxDimension * sizeof(float), 64);
  return tile_point_mat;
}

static void* get_tile_dist_mat() {
  if (!tile_dist_mat)
    tile_dist_mat = (uint8_t*)_mm_malloc(
      kMaxTileSize * kMaxLeaderSize * sizeof(float), 64);
  return tile_dist_mat;
}


template <class PR, class Bucket>
void assign_leaders(PR& Points,
                    const Bucket& ids,
                    const Bucket& leaders,
                    size_t fanout,
                    parlay::sequence<std::pair<uint32_t,uint32_t>>& flat,
                    int granularity) {
  using coord_type = typename PR::Point::T;
  const size_t N   = ids.size();
  const size_t M   = leaders.size();
  const size_t dim = Points.dimension();
#ifdef LOWMEM_INT8_L2
  if constexpr (std::is_same_v<coord_type,int8_t>||PipNNSignedCarrier<typename PR::Point>::value) {
    // Native signed-byte distances; preserve packed score/leader-ordinal ties
    // and chunk scheduling without FP32 matrices or T*M GEMM scratch.
    auto integer_chunk = [&](size_t chunk_idx) {
      size_t start=chunk_idx*kMaxTileSize, end=std::min(start+kMaxTileSize,N);
      std::vector<double> codes(M);
      for(size_t ti=start;ti<end;++ti) {
        for(size_t j=0;j<M;++j)
          codes[j]=packFloatAndInt(Points[ids[ti]].distance(Points[leaders[j]]),j);
        VQPartialSort(codes.data(),codes.data()+M,fanout);
        for(size_t fn=0;fn<fanout;++fn) {
          auto pair=unpackDouble(codes[fn]);
          flat[ti*fanout+fn]=std::make_pair(pair.second,ids[ti]);
        }
      }
    };
    size_t chunks=(N+kMaxTileSize-1)/kMaxTileSize;
    if(N<static_cast<size_t>(granularity))for(size_t c=0;c<chunks;++c)integer_chunk(c);
    else parlay::parallel_for(0,chunks,[&](size_t c){integer_chunk(c);});
    return;
  }
#endif

  //assert(M   <= kMaxLeaderSize);
  //assert(dim <= kMaxDimension);

  // this is a buffer for the leader points shared across all threads spawned here
  float* Lmat = (float*)_mm_malloc(M * dim * sizeof(float), 64);
  for (size_t j = 0; j < M; ++j) {
    coord_type* loc = reinterpret_cast<coord_type*>(Points.location(leaders[j]));
    for (size_t d = 0; d < dim; ++d) {
        Lmat[j*dim + d] = static_cast<float>(loc[d]);
    }
}
  //std::cout << "leader buffer allocated" << std::endl;
  // eigen map leader points
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> L(Lmat, M, dim);
  // compute square norms of leaders
  Eigen::Matrix<float, Eigen::Dynamic, 1> leader_sq = L.rowwise().squaredNorm();

  // routine to do matrix multiplication and selection
  auto process_chunk = [&](size_t chunk_idx) {
    size_t start = chunk_idx * kMaxTileSize;
    size_t end   = std::min(start + kMaxTileSize, N);
    size_t T     = end - start;

    // copy queries into thread-local buffer
    float* Pmat = pipnn_clean_routing_points(T * dim);
    for (size_t ti = 0; ti < T; ++ti) {
      auto loc = (coord_type*)Points.location(ids[start + ti]);
      for (size_t d = 0; d < dim; ++d)
        Pmat[ti*dim + d] = static_cast<float>(loc[d]);
    }

    // map P and the output D
    Eigen::Map<
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    > P(Pmat, T, dim),
      D(pipnn_clean_routing_distances(T * M), T, M);

    // GEMM + finish squared distances
    D.noalias() = P * L.transpose();
    Eigen::Matrix<float, Eigen::Dynamic,1> P_sq = P.rowwise().squaredNorm();


    using Point = typename PR::Point;

    if (Point::is_metric()) {
    // 1) scale the dot‐products by -2
    D *= -2.0f;

    // 2) add each query’s norm to its entire row:
    //    P_sq is a (T×1) vector, so colwise+= broadcasts it to every column
    D.colwise() += P_sq;

    // 3) add each leader’s norm to its entire column:
    //    leader_sq is (M×1), so we need it as a RowVector (1×M) to rowwise+=
    D.rowwise() += leader_sq.transpose();
    }
    else {
    // non‐metric: just negate the raw dot‐products
    D = -D;
    }

    std::vector<double> cc;
    cc.resize(M);
    for (size_t ti = 0; ti < T; ++ti) {
      float* row = D.data() + ti * M;
      for (size_t j =0; j < M; ++j) {
        PIPNN_PRECISION_RECORD(10,ids[start+ti],ids[start+ti],leaders[j],0,row[j],0,true);
        cc[j] = packFloatAndInt(row[j],j);
      }
      double* cd = (double*)cc.data();
      VQPartialSort(cd, cd + cc.size(), fanout);
      std::pair<float, uint32_t>* pairs = (std::pair<float, uint32_t>*)cd;
      for (size_t fn = 0; fn < fanout; ++fn) {
        double d = cd[fn];
        pairs[fn] = unpackDouble(d);
        flat[(start + ti)*fanout + fn] = std::make_pair(pairs[fn].second, ids[start + ti]);
      }
    }
  };

  size_t num_chunks = (N + kMaxTileSize - 1) / kMaxTileSize;
  if (N < (size_t)granularity) {
    for (size_t c = 0; c < num_chunks; ++c) process_chunk(c);
  } else {
    parlay::parallel_for(0, num_chunks,
      [&](size_t c){ process_chunk(c); }
    );
  }

  _mm_free(Lmat);
}


template<typename GraphH, typename PR>
void eigen_knn(GraphH &G, parlay::sequence<uint32_t> &active_indices, PR &Points, size_t MST_DEG) {
  auto N = active_indices.size();
  // Exact symmetric block-pair leaf kNN. Every unordered point pair is still
  // evaluated once, but no worker materializes an N x N distance matrix.
  using indexType = uint32_t;
  using coord_type = typename PR::Point::T;
  const size_t dimension = Points.get_dims();
  constexpr size_t B = 128;
  const size_t leaf_candidates=MST_DEG;
  float* point_mat = pipnn_clean_leaf_points(N * dimension);
  for (size_t i = 0; i < N; ++i) {
    coord_type* src = reinterpret_cast<coord_type*>(Points.location(active_indices[i]));
    for (size_t d = 0; d < dimension; ++d)
      point_mat[i * dimension + d] = static_cast<float>(src[d]);
  }
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
      Eigen::RowMajor>> P(point_mat, N, dimension);
  Eigen::VectorXf norms = P.rowwise().squaredNorm();
  std::vector<float> best_dist(N * leaf_candidates,
      std::numeric_limits<float>::infinity());
  std::vector<uint32_t> best_id(N * leaf_candidates,
      std::numeric_limits<uint32_t>::max());
  auto offer = [&](size_t row, uint32_t id, float dist) {
    size_t pos = leaf_candidates;
    for (size_t k = 0; k < leaf_candidates; ++k) {
      PIPNN_PRECISION_RECORD(6, active_indices[row], active_indices[row], active_indices[id], best_id[row*leaf_candidates+k] == std::numeric_limits<uint32_t>::max() ? UINT32_MAX : active_indices[best_id[row*leaf_candidates+k]], dist, best_dist[row*leaf_candidates+k], dist < best_dist[row*leaf_candidates+k] || (dist == best_dist[row*leaf_candidates+k] && id < best_id[row*leaf_candidates+k]), k);
      if (dist < best_dist[row * leaf_candidates + k] ||
          (dist == best_dist[row * leaf_candidates + k] &&
           id < best_id[row * leaf_candidates + k])) { pos = k; break; }
    }
    if (pos == leaf_candidates) return;
    for (size_t k = leaf_candidates - 1; k > pos; --k) {
      best_dist[row * leaf_candidates + k] = best_dist[row * leaf_candidates + k - 1];
      best_id[row * leaf_candidates + k] = best_id[row * leaf_candidates + k - 1];
    }
    best_dist[row * leaf_candidates + pos] = dist;
    best_id[row * leaf_candidates + pos] = id;
  };
  for (size_t ib = 0; ib < N; ib += B) {
    const size_t ni = std::min(B, N - ib);
    for (size_t jb = 0; jb <= ib; jb += B) {
      const size_t nj = std::min(B, N - jb);
      float* tile = pipnn_clean_leaf_distances(ni * nj);
      Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
          Eigen::RowMajor>> D(tile, ni, nj);
#ifdef LOWMEM_INT8_L2
      if constexpr (!std::is_same_v<coord_type,int8_t>&&!PipNNSignedCarrier<typename PR::Point>::value)
#endif
      D.noalias() = P.middleRows(ib, ni) * P.middleRows(jb, nj).transpose();
      for (size_t ii = 0; ii < ni; ++ii) {
        const size_t i = ib + ii;
        const size_t jlimit = (ib == jb) ? ii : nj;
        for (size_t jj = 0; jj < jlimit; ++jj) {
          const size_t j = jb + jj;
          float dist;
#ifdef LOWMEM_INT8_L2
          if constexpr (std::is_same_v<coord_type,int8_t>||PipNNSignedCarrier<typename PR::Point>::value)
            dist = Points[active_indices[i]].distance(Points[active_indices[j]]);
          else
#endif
          dist = PR::Point::is_metric()
              ? -2.0f * D(ii, jj) + norms[i] + norms[j]
              : -D(ii, jj);
          PIPNN_PRECISION_RECORD(9, active_indices[i], active_indices[i], active_indices[j], 0, dist, 0, true);
          offer(i, static_cast<uint32_t>(j), dist);
          offer(j, static_cast<uint32_t>(i), dist);
        }
      }
    }
  }
  for (size_t i = 0; i < N; ++i) {
    const uint32_t global_i = active_indices[i];
    for (size_t k = 0; k < MST_DEG; ++k) {
      if (best_id[i * MST_DEG + k] == std::numeric_limits<uint32_t>::max()) continue;
      const uint32_t global_j = active_indices[best_id[i * MST_DEG + k]];
      const float dist = best_dist[i * MST_DEG + k];
      std::pair<uint32_t, float> forward = {global_j, dist};
      std::pair<uint32_t, float> reverse = {global_i, dist};
      G.merge_neighbors(global_i, absl::MakeSpan(&forward, 1));
      G.merge_neighbors(global_j, absl::MakeSpan(&reverse, 1));
    }
  }
  return;
}


} // end namespace

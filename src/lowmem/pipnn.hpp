#pragma once

#include <math.h>

#include <algorithm>
#include <queue>
#include <cstdlib>
#include <random>
#include <set>
#include <utility>
#include <bitset>

#include "../../third_party/pipnn/include/bench/parse_command_line.h"
#include "../../third_party/pipnn/include/utils/graph.h"
#include "../../third_party/pipnn/include/PipNN/memory_clean_workspace.h"
#include "b64_graph.hpp"
#ifdef TW_REPRESENTATIVE_ORACLES
#include "../../third_party/pipnn/include/PipNN/tw_representative_oracles.h"
#endif
#include "../../third_party/pipnn/include/PipNN/pipnn_utils.h"
#include "../../third_party/pipnn/include/PipNN/b64_b96_diagnostic.h"
#include "../../third_party/parlay/include/parlay/parallel.h"
#include "../../third_party/parlay/include/parlay/primitives.h"
#include "../../third_party/parlay/include/parlay/random.h"


namespace parlayANN {

std::atomic<size_t> leaf_count(0);

template <typename GraphH, typename Point, typename PointRange, typename indexType>
struct cluster {
  using distanceType = typename Point::distanceType;
  using edge = std::pair<indexType, indexType>;
  using labelled_edge = std::pair<edge, distanceType>;
  using GraphI = Graph<indexType>;
  using PR = PointRange;
  using Bucket = parlay::sequence<uint32_t>;


  struct CompareBucketsBySize {
    bool operator()(const Bucket &lhs, const Bucket &rhs) const {
      return lhs.size() > rhs.size();
    }
  };

  using utils = cluster_utils<Point, PointRange, indexType>;

  cluster() {}

  // ===== Shared helpers used by both LevelByLevelSketch and RecursivelySketch =====

  // Selects leaders by random sampling, assigns all points in `ids` to their
  // nearest leaders, and returns the resulting non-empty clusters.
  parlay::sequence<Bucket> SelectAndAssignLeaders(
      PR& Points, Bucket& ids, int depth, int initial_fanout,
      size_t seed, int fanout_levels, std::vector<int>& fanout_scheme,
      size_t fixed_leaders
      ) {
    size_t num_leaders =
        (depth == 0) ? TOP_LEVEL_NUM_LEADERS : ids.size() * FRACTION_LEADERS;
    if (fixed_leaders != 0) num_leaders = fixed_leaders;
    num_leaders = std::min<size_t>(num_leaders, MAX_NUM_LEADERS);
    num_leaders = std::max<size_t>(num_leaders, 3);

    COMPARISONS.fetch_add(num_leaders * ids.size(), std::memory_order_relaxed);

    Bucket leaders;
    std::mt19937 prng(seed);
    leaders.resize(num_leaders);
    std::sample(ids.begin(), ids.end(), leaders.begin(), leaders.size(), prng);

    int fanout = initial_fanout;
    if (depth < (int)fanout_scheme.size()) {
      fanout = fanout_scheme[depth];
    }
    fanout = std::min<int>(fanout, (int)num_leaders);

    int granularity = 5000;
    parlay::internal::timer t;
    t.start();
    parlay::sequence<std::pair<uint32_t, uint32_t>> flat(ids.size() * fanout);
    assign_leaders(Points, ids, leaders, fanout, flat, granularity);
    if (depth == 0) {
      t.next("first level assign time");
    }

    parlay::sequence<Bucket> clusters =
        parlay::group_by_index(flat, leaders.size());
    leaders.clear();

    // Filter out empty clusters produced by group_by_index.
    clusters = parlay::filter(
        clusters, [](const Bucket& b) { return b.size() > 0; });
    return clusters;
  }

  // Merges undersized clusters using a priority queue. Separates the input
  // clusters into:
  //   - finished_buckets: properly sized buckets (done, no further processing)
  //   - remaining_clusters: buckets that still need recursion or further splitting
  // When NO_MERGE is true, all clusters go directly to remaining_clusters.
  void MergeBuckets(
      parlay::sequence<Bucket>& clusters,
      std::vector<Bucket>& finished_buckets,
      parlay::sequence<Bucket>& remaining_clusters) {
    if (NO_MERGE) {
      remaining_clusters = std::move(clusters);
      return;
    }

    std::priority_queue<Bucket, std::vector<Bucket>, CompareBucketsBySize> pq;
    for (size_t i = 0; i < clusters.size(); ++i) {
      pq.push(std::move(clusters[i]));
    }
    clusters.resize(0);

    while (pq.size() > 1) {
      auto& top = pq.top();
      if (top.size() >= MIN_CLUSTER_SIZE) break;

      size_t merged_size = top.size();
      std::vector<Bucket> merged_buckets;
      merged_buckets.emplace_back(top);
      pq.pop();

      while (!pq.empty() && merged_size < MIN_CLUSTER_SIZE &&
             merged_size + pq.top().size() <= MAX_MERGED_CLUSTER_SIZE) {
        merged_size += pq.top().size();
        merged_buckets.emplace_back(pq.top());
        pq.pop();
      }

      if (merged_buckets.size() == 1) {
        CHECK(merged_buckets[0].size() <= MAX_CLUSTER_SIZE);
        if (merged_buckets[0].size() > MAX_CLUSTER_SIZE) {
          std::cout << "Pushing a cluster that is too large: "
                    << merged_buckets[0].size() << std::endl;
        }
        finished_buckets.emplace_back(std::move(merged_buckets[0]));
        break;
      }

      CHECK(merged_size > 0);
      auto new_bucket =
          parlay::sequence<uint32_t>::uninitialized(merged_size);
      size_t pos = 0;
      for (auto& b : merged_buckets) {
        std::memcpy(new_bucket.begin() + pos, b.begin(),
                    b.size() * sizeof(uint32_t));
        pos += b.size();
      }
      new_bucket = parlay::remove_duplicates(new_bucket);
      CHECK(new_bucket.size() > 0);

      if (new_bucket.size() >= MIN_CLUSTER_SIZE &&
          new_bucket.size() <= MAX_CLUSTER_SIZE) {
        finished_buckets.push_back(std::move(new_bucket));
      } else {
        pq.push(std::move(new_bucket));
      }
    }

    while (!pq.empty()) {
      remaining_clusters.emplace_back(pq.top());
      pq.pop();
    }
  }


  // Base-case handler for duplicates / near-duplicates. When recursion depth
  // is too deep or shrinkage ratio is too small, randomly shuffle and split
  // the bucket into MAX_CLUSTER_SIZE-sized pieces.
  void SplitDuplicates(
      Bucket& bucket, size_t seed, int depth, size_t parent_size,
      std::vector<Bucket>& output_buckets) {
    auto ids_copy = std::move(bucket);
    std::mt19937 prng(seed + depth + parent_size);
    std::shuffle(ids_copy.begin(), ids_copy.end(), prng);

    size_t full_blocks = ids_copy.size() / MAX_CLUSTER_SIZE;
    size_t remainder = ids_copy.size() - full_blocks * MAX_CLUSTER_SIZE;

    if (full_blocks > 0 && remainder < MIN_CLUSTER_SIZE) {
      full_blocks--;
      if (full_blocks > 0) {
        for (size_t k = 0; k < full_blocks; ++k) {
          auto& new_bucket = output_buckets.emplace_back();
          size_t start = MAX_CLUSTER_SIZE * k;
          size_t end = std::min(start + MAX_CLUSTER_SIZE, ids_copy.size());
          for (size_t j = start; j < end; ++j) new_bucket.push_back(ids_copy[j]);
          CHECK(new_bucket.size() <= MAX_CLUSTER_SIZE);
        }
      }
      size_t start = full_blocks * MAX_CLUSTER_SIZE;
      size_t end = ids_copy.size();
      size_t block_size = (end - start) / 2;
      if (block_size < MIN_CLUSTER_SIZE) {
        std::cout << "Violated expectation." << std::endl;
        exit(-1);
      }
      auto& new_bucket = output_buckets.emplace_back();
      for (size_t j = start; j < start + block_size; ++j)
        new_bucket.push_back(ids_copy[j]);
      CHECK(new_bucket.size() <= MAX_CLUSTER_SIZE);
      auto& next_bucket = output_buckets.emplace_back();
      for (size_t j = start + block_size; j < end; ++j)
        next_bucket.push_back(ids_copy[j]);
      CHECK(next_bucket.size() <= MAX_CLUSTER_SIZE);
    } else {
      for (size_t i = 0; i < ids_copy.size(); i += MAX_CLUSTER_SIZE) {
        auto& new_bucket = output_buckets.emplace_back();
        size_t start = i;
        size_t end = std::min(start + MAX_CLUSTER_SIZE, ids_copy.size());
        for (size_t j = start; j < end; ++j) new_bucket.push_back(ids_copy[j]);
      }
    }
  }

  // ===== Main sketch functions =====

  // Returns a collection of leaf buckets (iterative / BFS style).
  std::vector<Bucket> LevelByLevelSketch(
      GraphH &G,
      PR &Points, std::vector<Bucket> initial_buckets, long cluster_size,
      int initial_fanout, size_t seed, int fanout_levels,
      std::vector<int> &fanout_scheme, size_t fixed_leaders) {
    std::vector<Bucket> active_buckets = std::move(initial_buckets);
    std::vector<Bucket> finished_buckets;
    int depth = 0;

    while (!active_buckets.empty()) {
      std::vector<Bucket> to_process;
      for (auto& b : active_buckets) {
        if (b.size() <= (size_t)cluster_size) {
          finished_buckets.push_back(std::move(b));
        } else {
          to_process.push_back(std::move(b));
        }
      }
      active_buckets.clear();
      if (to_process.empty()) break;

      parlay::sequence<std::vector<Bucket>> next_level_results(to_process.size());

      parlay::parallel_for(0, to_process.size(), [&](size_t i) {
        Bucket& ids = to_process[i];
        std::vector<Bucket>& result_buckets = next_level_results[i];

        size_t current_seed = seed + depth * 1337 + i * 7;
        int effective_fanout = (depth >= fanout_levels) ? 1 : initial_fanout;

        parlay::sequence<Bucket> clusters = SelectAndAssignLeaders(
            Points, ids, depth, effective_fanout, current_seed,
            fanout_levels, fanout_scheme, fixed_leaders);

        std::vector<Bucket> merged_finished;
        parlay::sequence<Bucket> remaining;
        MergeBuckets(clusters, merged_finished, remaining);

        // Classify each output bucket.
        for (auto& b : merged_finished) {
          result_buckets.push_back(std::move(b));
        }
        for (auto& b : remaining) {
          if (depth > MAX_DEPTH ||
              (depth > CONCERNING_DEPTH &&
               b.size() > TOO_SMALL_SHRINKAGE_FRACTION * ids.size())) {
            SplitDuplicates(b, seed, depth, ids.size(), result_buckets);
          } else {
            result_buckets.push_back(std::move(b));
          }
        }
      });

      // Flatten results into active_buckets for the next level.
      for (size_t i = 0; i < next_level_results.size(); ++i) {
        for (auto& b : next_level_results[i]) {
          active_buckets.push_back(std::move(b));
        }
      }

      std::cout << "Processed level " << depth << std::endl;
      depth++;
      size_t sum_size = 0;
      size_t max_size = 0;
      size_t min_size = active_buckets.empty()
                            ? 0
                            : std::numeric_limits<size_t>::max();
      for (const auto& b : active_buckets) {
        sum_size += b.size();
        if (b.size() > max_size) max_size = b.size();
        if (b.size() < min_size) min_size = b.size();
      }
      std::cout << "Level " << depth
                << " active clusters: " << active_buckets.size()
                << " sum: " << sum_size << " max: " << max_size
                << " min: " << min_size << " avg: "
                << (active_buckets.empty()
                        ? 0
                        : (double)sum_size / active_buckets.size())
                << std::endl;

      if (depth > 100) break;
    }
    for (auto& b : active_buckets)
      finished_buckets.push_back(std::move(b));

    return finished_buckets;
  }

  // Returns a collection of leaf buckets (recursive / DFS style).
  std::vector<Bucket> RecursivelySketch(
      GraphH &G,
      PR &Points, Bucket &ids, long cluster_size, int depth, int fanout,
      size_t seed, int fanout_levels, std::vector<int> &fanout_scheme,
      size_t fixed_leaders) {
    if (ids.size() <= (size_t)cluster_size) {
      return {ids};
    }

    size_t next_seed = parlay::hash64(seed);

    parlay::sequence<Bucket> clusters = SelectAndAssignLeaders(
        Points, ids, depth, fanout, seed, fanout_levels, fanout_scheme,
        fixed_leaders);

    // Separate into finished buckets and clusters needing further recursion.
    std::vector<Bucket> buckets;
    parlay::sequence<Bucket> remaining;
    MergeBuckets(clusters, buckets, remaining);

    // Recurse on remaining clusters.
    parlay::sequence<std::vector<Bucket>> rec_buckets(remaining.size());
    parlay::parallel_for(
        0, remaining.size(),
        [&](size_t cluster_id) {
          std::vector<Bucket> recursive_buckets;
          if (depth > MAX_DEPTH ||
              (depth > CONCERNING_DEPTH &&
               remaining[cluster_id].size() >
                   TOO_SMALL_SHRINKAGE_FRACTION * ids.size())) {
            SplitDuplicates(remaining[cluster_id], seed, depth, ids.size(),
                            recursive_buckets);
          } else {
            auto local_fanout = (depth < fanout_levels) ? fanout : 1;
            recursive_buckets = RecursivelySketch(
                G, Points, remaining[cluster_id], cluster_size, depth + 1,
                /*fanout=*/local_fanout, next_seed + cluster_id, fanout_levels,
                fanout_scheme, fixed_leaders);
          }
          rec_buckets[cluster_id] = std::move(recursive_buckets);
        },
        1);

    for (size_t cluster_id = 0; cluster_id < remaining.size(); ++cluster_id) {
      while (!rec_buckets[cluster_id].empty()) {
        size_t index = rec_buckets[cluster_id].size();
        buckets.push_back(std::move(rec_buckets[cluster_id][index - 1]));
        rec_buckets[cluster_id].pop_back();
      }
    }

    return buckets;
  }

  // Depth-first partition-and-consume. Completed leaves never enter a global
  // materialized bucket vector: their proposal stream is reduced immediately.
  // This preserves the exact leaf multiset and only changes its consumption
  // schedule (HashPrune/WARD is a history-independent set reduction).
  void RecursivelySketchAndConsume(
      GraphH& G, PR& Points, Bucket& ids, long cluster_size, int depth,
      int fanout, size_t seed, int fanout_levels,
      std::vector<int>& fanout_scheme, size_t fixed_leaders,
      uint64_t oracle_path = 0
      ) {
    if (ids.size() <= static_cast<size_t>(cluster_size)) {
      streamed_memberships.fetch_add(ids.size(), std::memory_order_relaxed);
#ifdef TW_REPRESENTATIVE_ORACLES
      const uint32_t oracle_leaf = tw_oracles().leaf(oracle_path, ids.begin(), ids.end());
      tw_oracles().set_current_leaf(oracle_leaf);
#endif
      if (ids.size() > 1) RunLeaf(G, Points, ids);
#ifdef TW_REPRESENTATIVE_ORACLES
      tw_oracles().clear_current_leaf();
#endif
      return;
    }
    const size_t next_seed = parlay::hash64(seed);
    auto clusters = SelectAndAssignLeaders(
        Points, ids, depth, fanout, seed, fanout_levels, fanout_scheme,
        fixed_leaders
        );
    const size_t parent_size = ids.size();
    std::vector<Bucket> finished;
    parlay::sequence<Bucket> remaining;
    MergeBuckets(clusters, finished, remaining);
    for (size_t finished_id = 0; finished_id < finished.size(); ++finished_id) {
      auto& b = finished[finished_id];
      streamed_memberships.fetch_add(b.size(), std::memory_order_relaxed);
#ifdef TW_REPRESENTATIVE_ORACLES
      const uint32_t oracle_leaf = tw_oracles().leaf(
          tw_path_child(oracle_path, depth, finished_id), b.begin(), b.end());
      tw_oracles().set_current_leaf(oracle_leaf);
#endif
      if (b.size() > 1) {
        RunLeaf(G, Points, b);
      }
#ifdef TW_REPRESENTATIVE_ORACLES
      tw_oracles().clear_current_leaf();
#endif
    }
    // Children own their membership now; retaining the parent across their
    // complete recursion only widens the live hypergraph frontier.
    ids = Bucket();
    auto consume_child = [&](size_t cluster_id) {
      auto& b = remaining[cluster_id];
#if defined(TW_REPRESENTATIVE_ORACLES) || defined(PIPNN_OVERLAP_COVER_ORACLE) || defined(PIPNN_OVERLAP_COVER_REPLAY) || defined(PIPNN_SOURCE_ROW_REPLAY) || defined(PIPNN_HIERARCHICAL_CONTRACTION_TRACE)
      // Finished and recursive children share a parent but occupy disjoint
      // canonical child labels in the oracle tree.
      const size_t oracle_child_id = finished.size() + cluster_id;
#endif
      if (depth > MAX_DEPTH ||
          (depth > CONCERNING_DEPTH &&
           b.size() > TOO_SMALL_SHRINKAGE_FRACTION * parent_size)) {
        std::vector<Bucket> split;
        SplitDuplicates(b, seed, depth, parent_size, split);
        for (auto& leaf : split) {
          streamed_memberships.fetch_add(leaf.size(), std::memory_order_relaxed);
#ifdef TW_REPRESENTATIVE_ORACLES
          const uint32_t oracle_leaf = tw_oracles().leaf(
              tw_path_child(oracle_path, depth, oracle_child_id),
              leaf.begin(), leaf.end());
          tw_oracles().set_current_leaf(oracle_leaf);
#endif
          if (leaf.size() > 1) {
            RunLeaf(G, Points, leaf);
          }
#ifdef TW_REPRESENTATIVE_ORACLES
          tw_oracles().clear_current_leaf();
#endif
        }
      } else {
        const auto local_fanout = (depth < fanout_levels) ? fanout : 1;
        RecursivelySketchAndConsume(
            G, Points, b, cluster_size, depth + 1, local_fanout,
            next_seed + cluster_id, fanout_levels, fanout_scheme,
            fixed_leaders
#if defined(TW_REPRESENTATIVE_ORACLES)
            , tw_path_child(oracle_path, depth, oracle_child_id)
#endif
            );
      }
      b = Bucket();
    };
    if (depth == 0) {
      // A bounded number of independent root subtrees keeps all workers busy,
      // while the barrier provides a safe point to release dead Parlay pools.
      // Each root child recursively allocates assignment and grouped-membership
      // arrays.  Running hundreds of children concurrently makes those exact,
      // phase-local arrays overlap the already-live candidate table.  Bound the
      // outer working set while retaining parallelism inside every subtree.
      // The environment override is an execution policy, not an ANN parameter:
      // it changes neither leaves nor their proposal multiset.
      size_t batch = std::min<size_t>(128, parlay::num_workers());
      if (const char* value = std::getenv("PIPNN_ROOT_SUBTREE_BATCH")) {
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(value, &end, 10);
        if (end != value && parsed > 0) batch = parsed;
      }
      batch = std::max<size_t>(1, std::min(batch, remaining.size()));
      std::cout << "Bounded root-subtree working set: " << batch << '\n';
      for (size_t begin = 0; begin < remaining.size(); begin += batch) {
        const size_t end = std::min(remaining.size(), begin + batch);
        parlay::parallel_for(begin, end, consume_child, 1);
        pipnn_clean_trim_allocator();
        if (end < remaining.size()) G.allocate_stage(begin / batch);
      }
    } else {
      parlay::parallel_for(0, remaining.size(), consume_child, 1);
    }
  }


  auto recursively_sketch_wrapper(GraphH &G, PR &Points, size_t cluster_size) {
    // TODO: this is non-deterministic! wtf.
    // https://en.cppreference.com/w/cpp/numeric/random/random_device.html
    std::random_device rd;
    const char* seed_text = std::getenv("PIPNN_BUILD_SEED");
    const uint32_t build_seed = seed_text
        ? static_cast<uint32_t>(std::stoul(seed_text)) : rd();
    std::mt19937 rng(build_seed);
    if (seed_text) std::cout << "PIPNN_BUILD_SEED=" << build_seed << '\n';
    std::uniform_int_distribution<int> uni(0, Points.size());
    parlay::random rnd(uni(rng));
    auto ids = parlay::tabulate(Points.size(), [&](uint32_t i) { return i; });
    parlay::internal::timer t;

    int fanout_per = FANOUT_PER_LEVEL;
    int fanout_levels = 0;
    auto local_fanout = FANOUT;
    std::cout << "FANOUT = " << FANOUT
              << " FANOUT_PER_LEVEL = " << FANOUT_PER_LEVEL << std::endl;

    if (fanout_per != 0) {
      // identify the number of levels to fanout at the per_level_fanout without
      // exceeding fanout. we are using a constant rate of fanout, which may not
      // be ideal. For power users, we may want to allow custom fanout
      // distributions. there may also be some way of dynamically adjusting the
      // fanout within the RecursivelySketch method based on distributional
      // properties of the data.
      local_fanout = fanout_per;
      int fprod = fanout_per;
      while (fprod * fanout_per <= FANOUT) {
        fprod *= fanout_per;
        fanout_levels++;
      }
    }
    std::cout << "number of levels to fanout: " << fanout_levels + 1
              << std::endl;
    std::cout << "fanout per level: " << local_fanout << std::endl;
    std::cout << "total fanout: " << std::pow(local_fanout, (fanout_levels + 1))
              << std::endl;
    std::cout << "total fanout allowance: " << FANOUT << std::endl;
    std::cout << "leader method: random" << std::endl;

    t.start();

    if (FIXED_NUM_LEADERS != 0) {
      std::cout << "Using fixed number of leaders: " << FIXED_NUM_LEADERS
                << std::endl;
    }

    std::vector<Bucket> buckets;
    if (!LEVEL_BY_LEVEL) {
      streamed_memberships.store(0, std::memory_order_relaxed);
#ifdef TW_REPRESENTATIVE_ORACLES
      tw_oracles().initialize(Points.size());
#endif
      RecursivelySketchAndConsume(
          G, Points, ids, cluster_size, 0, local_fanout, SEED, fanout_levels,
          FANOUT_SCHEME, FIXED_NUM_LEADERS, 0
          );
      SEED = parlay::hash64(SEED);
      t.next("partition-and-consume time");
      std::cout << "Streamed leaf memberships: "
                << streamed_memberships.load(std::memory_order_relaxed) << '\n';
      pipnn_clean_report_memory("after_streaming_leaves");
      pipnn_clean_report_workspace_bytes("after_streaming_leaves");
      G.report_budget_phase("after_streaming_leaves");
      pipnn_clean_release_routing_workspaces();
      pipnn_clean_release_leaf_workspaces();
      return;
    }
    if (LEVEL_BY_LEVEL) {
        std::vector<Bucket> initial_buckets;
        initial_buckets.push_back(std::move(ids));
        std::cout << "Calling LevelByLevelSketch" << std::endl;
        buckets = LevelByLevelSketch(
            G, Points, initial_buckets, cluster_size, local_fanout, SEED, fanout_levels,
            FANOUT_SCHEME, FIXED_NUM_LEADERS);
    } else {
        buckets = RecursivelySketch(
            G, Points, ids, cluster_size, 0, local_fanout, SEED, fanout_levels,
            FANOUT_SCHEME, FIXED_NUM_LEADERS);
    }

    SEED = parlay::hash64(SEED);
    t.next("buckets time");
    //get sizeof buckets in bytes
    size_t total_bytes = 0;
    size_t total_points = 0;
    for (size_t i = 0; i < buckets.size(); i++) {
      total_bytes += sizeof(buckets[i]);
      total_bytes += buckets[i].size() * sizeof(uint32_t);
      total_points += buckets[i].size();
    }
    std::cout << "buckets size (bytes): " << total_bytes << std::endl;
    std::cout << "Total buckets: " << buckets.size() << std::endl;
    std::cout << "Total points: " << total_points << std::endl;

    std::cout << "Computed buckets!" << std::endl;
    pipnn_clean_report_memory("before_routing_workspace_release");
    pipnn_clean_report_workspace_bytes("before_routing_workspace_release");
    pipnn_clean_release_routing_workspaces();

    //G.allocate_graph(Points.size());
    // Build on each bucket. // note the hack of checking size. This is a hack
    // to get around the RecursivelySketch routine producing empty clusters...
    // Looking into it.
    std::cout << "Building leaves:" << std::endl;
     parlay::parallel_for(0, buckets.size(), [&](size_t i) {
       CHECK(buckets[i].size() <= MAX_CLUSTER_SIZE);
       if (buckets[i].size() > 1) RunLeaf(G, Points, buckets[i]);
       // The reducer has consumed this leaf completely. Its membership list is
       // never consulted again, so hand the storage back while other leaves
       // are still being processed instead of retaining all overlap to phase end.
       buckets[i] = Bucket();
       // Note we can still have some small clusters due to how sampling works.
     });
    pipnn_clean_report_memory("before_leaf_workspace_release");
    pipnn_clean_report_workspace_bytes("before_leaf_workspace_release");
    t.next("build leaf time");
    pipnn_clean_release_leaf_workspaces();
  }

  void multiple_clustertrees(GraphH &G, PR &Points, long cluster_size,
                             long num_clusters) {
    G.allocate_graph(Points.size());
    for (auto i = 0; i < num_clusters; i++) {
      recursively_sketch_wrapper(G, Points, cluster_size);
      pipnn_clean_trim_allocator();
      pipnn_clean_report_memory("after_leaf_round");
      G.report_budget_phase("after_leaf_round");
      std::cout << "Built cluster " << i << " of " << num_clusters << std::endl;
      std::cout << "Leaf count: " << leaf_count << std::endl;
      std::cout << "# of comparisons during clustering: " << COMPARISONS
                << std::endl;
    }
  }

  /*  Leaf code  */
  template <class LeafGraph>
  void RunLeaf(LeafGraph &G, PR &Points,
               parlay::sequence<uint32_t> &active_indices) {
    leaf_count++;
    EigenKNN(G, Points, active_indices);
  }

  template <class LeafGraph>
  void EigenKNN(LeafGraph &G, PR &Points,
                     parlay::sequence<uint32_t> &active_indices) {
    eigen_knn(G, active_indices, Points, MST_DEG);
  }

  size_t SEED = 555;
  double FRACTION_LEADERS = 0.005;
  size_t TOP_LEVEL_NUM_LEADERS = 950;
  size_t MAX_NUM_LEADERS = 512;
  size_t MAX_CLUSTER_SIZE = 5000;
  size_t MIN_CLUSTER_SIZE = 1;
  size_t MAX_MERGED_CLUSTER_SIZE = 2500;
  size_t FIXED_NUM_LEADERS = 0; //0 means use fraction
  int MAX_DEPTH = 3; // 8
  int CONCERNING_DEPTH = 2;  // 6
  double TOO_SMALL_SHRINKAGE_FRACTION = 0.8;
  size_t MST_DEG = 3;
  int FANOUT = 1;
  int FANOUT_PER_LEVEL = 1;
  bool LEVEL_BY_LEVEL = false;
  bool NO_MERGE = false;
  double ALPHA = 1;

  std::vector<int> FANOUT_SCHEME = {};
  std::atomic<size_t> COMPARISONS = 0;
  std::atomic<size_t> streamed_memberships = 0;
};


template <typename Point, typename PointRange, typename indexType>
struct pipnn_index {
  using GraphI = Graph<indexType>;
  using PR = PointRange;

  using utils = cluster_utils<Point, PointRange, indexType>;

  pipnn_index() {}

  template <typename GraphH>
  void ExportToGraph(GraphH& graph, commandLine& P, GraphI& G, PR& Points, bool prune_all, size_t prune_degree, double alpha) {
    parlay::internal::timer t;
    t.start();

    auto degree_seq = parlay::delayed_tabulate(Points.size(), [&] (size_t i) -> size_t {
      return graph.degree(i);
    });
    std::cout << "Average table size: " << (parlay::reduce(degree_seq) / Points.size()) << std::endl;

    lowmem_profile::mark("candidate_capture");graph.finalize();
    handoff_capture(graph, Points.size());
    handoff_pre_prepare(graph,Points.size());
    if(const char* path=std::getenv("PIPNN_CANDIDATE_GEOMETRY_OUT"))
      export_candidate_geometry(graph,Points.size(),path);

    // Read-only incoming candidate support, before RobustPrune/discard.
    if (const char* path = std::getenv("PIPNN_HUBNESS_SUPPORT_OUT")) {
      std::vector<uint32_t> support(Points.size(), 0);
      parlay::parallel_for(0, Points.size(), [&](size_t u) {
        for (auto candidate : graph.Neighbors(u))
          __atomic_fetch_add(&support[candidate.point_id], 1u, __ATOMIC_RELAXED);
      });
      std::ofstream out(path, std::ios::binary);
      if (!out || !out.write(reinterpret_cast<const char*>(support.data()), support.size()*4))
        throw std::runtime_error("hubness candidate support export failed");
    }

    if (const char* rows_path = std::getenv("PIPNN_CANDIDATE_ROWS_OUT")) {
      std::ofstream rows(rows_path);
      if (!rows) throw std::runtime_error("cannot open candidate row snapshot");
      size_t samples = 512;
      if (const char* count = std::getenv("PIPNN_CANDIDATE_ROWS_COUNT"))
        samples = std::max<size_t>(1, std::strtoull(count, nullptr, 10));
      for (size_t j = 0; j < samples; ++j) {
        const size_t i = j * Points.size() / samples;
        rows << i << ' ' << graph.degree(i);
        for (auto ngh : graph.Neighbors(i))
          rows << ' ' << ngh.point_id << ':' << static_cast<float>(ngh.norm);
        rows << '\n';
      }
    }

    if (const char* digest_path = std::getenv("PIPNN_AEA_CANDIDATE_DIGEST_OUT")) {
      b64_b96_write_candidate_digest(graph, Points.size(), digest_path);
    }





    // Two-Witness is a metric lune/RNG rule. MIPS exposes the non-metric score
    // -dot, for which both raw multiplicative tests and base-vector lifts
    // collapse on strongly co-directional embeddings. Preserve nearest WARD
    // evidence directly for non-metric point types; L2 TW remains unchanged.
    using witness_distance_type = typename Point::distanceType;
    constexpr witness_distance_type witness_distance_shift = 0;
    const double witness_alpha = alpha;
#ifdef TW_REPRESENTATIVE_ORACLES
    tw_oracles().pre_export(graph, Points, alpha);
#endif

    //create LSH table
    //auto H = SimHash<Point, PR, indexType>(12, Points.dimension());

#if defined(PIPNN_MIPS_WITNESS_ANGULAR) || \
    defined(PIPNN_MIPS_WITNESS_ANGULAR_NORM)
    // Similarity-aware MIPS witnesses need only one 4-byte norm per point;
    // normalized vectors would duplicate the entire dataset.  Compute once
    // and reuse throughout source-local finalization.
    std::vector<float> mips_point_norms(Points.size());
    parlay::parallel_for(0, Points.size(), [&](size_t i) {
      auto point = Points[i];
      double squared_norm = 0.0;
      for (long j = 0; j < Points.dimension(); ++j) {
        const double value = static_cast<double>(point[j]);
        squared_norm += value * value;
      }
      mips_point_norms[i] = static_cast<float>(std::sqrt(squared_norm));
    });
#endif

    // Optional construction-only selector trace.  It records the evidence
    // competition seen by TW before the in-place candidate state is retired.
    // No query statistics or alternative finalizer are involved.
    const char* tw_row_stats_path = std::getenv("PIPNN_TW_ROW_STATS_OUT");
    std::vector<uint16_t> tw_row_candidates, tw_row_final;
    if (tw_row_stats_path && *tw_row_stats_path) {
      tw_row_candidates.resize(Points.size());
      tw_row_final.resize(Points.size());
    }
    B64B96FinalDegrees diagnostic_degrees(Points.size());
    lowmem_profile::mark("initial_prune_stream");auto export_one = [&](size_t i) {
      // Exact source-local prune followed by overwrite of the source's own
      // 224-byte candidate base block. No separate final adjacency exists.
      thread_local std::vector<indexType> final_ids;
      if(bpre::mode){
        utils::robustPruneIds(bpre::pool(i,graph,Points),i,Points,alpha,
          std::min<size_t>(prune_degree,diagnostic_degrees.limit(i)),final_ids);
      }else {
      utils::robustPruneIds(graph.Neighbors(i), i, Points, alpha,
                            std::min<size_t>(prune_degree, diagnostic_degrees.limit(i)), final_ids);
      }
#ifdef TW_REPRESENTATIVE_ORACLES
      tw_oracles().node(i, graph, graph.Neighbors(i), Points, alpha, final_ids);
      const char* replay_mode = std::getenv("TW_LAMINAR_REPLAY");
      if (replay_mode && std::string(replay_mode) != "full") {
        auto replay_candidates = tw_oracles().replay(
            i, graph.Neighbors(i), final_ids);
        if (Point::is_metric())
          utils::robustPruneTwoWitnessIds(replay_candidates, i, Points,
                                          witness_alpha, prune_degree,
                                          final_ids, witness_distance_shift);
        else
          utils::nearestIds(replay_candidates, i, prune_degree, final_ids);
      }
#endif
      if (!tw_row_candidates.empty()) {
        tw_row_candidates[i] = static_cast<uint16_t>(graph.degree(i));
        tw_row_final[i] = static_cast<uint16_t>(final_ids.size());
      }
      diagnostic_degrees.record(i, final_ids.size());
      graph.write_final_row_r64(i, final_ids);
    };
    constexpr size_t kFinalizeChunk = 65536;
    for (size_t begin = 0; begin < Points.size(); begin += kFinalizeChunk) {
      const size_t end = std::min(Points.size(), begin + kFinalizeChunk);
      parlay::parallel_for(begin, end, export_one);
      graph.discard_candidate_rows(begin, end);
      pipnn_clean_report_memory("r64_finalize_chunk");
    }
    diagnostic_degrees.finish();
    if (!tw_row_candidates.empty()) {
      std::ofstream stats(tw_row_stats_path, std::ios::binary);
      if (!stats) throw std::runtime_error("cannot create TW row stats");
      const uint32_t magic = 0x54575231u;  // TWR1
      const uint64_t n = Points.size();
      stats.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
      stats.write(reinterpret_cast<const char*>(&n), sizeof(n));
      stats.write(reinterpret_cast<const char*>(tw_row_candidates.data()),
                  n * sizeof(uint16_t));
      stats.write(reinterpret_cast<const char*>(tw_row_final.data()),
                  n * sizeof(uint16_t));
      if (!stats) throw std::runtime_error("TW row stats write failed");
      std::cout << "TW_ROW_STATS rows=" << n << " path=" << tw_row_stats_path
                << std::endl;
    }
#ifdef TW_REPRESENTATIVE_ORACLES
    tw_oracles().finish(graph, prune_degree * sizeof(indexType));
#endif
    graph.handoff_r64_streaming(G);
    t.next("Final prune time");
  }

  void build_index(GraphI &G, commandLine& P, PR &Points, long cluster_rounds,
                   long cluster_size, long MSTDeg, bool prune,
                   bool prune_all, double alpha,
                   long prune_degree,
                   long top_level_leaders, int fanout,
                   int fanout_per_level, double fraction_leaders,
                   std::string fanout_scheme, int Hsize, long table_size,
                   bool no_reservoir, size_t fixed_num_leaders) {
    if (no_reservoir)
      throw std::runtime_error("AEA requires a bounded evidence reservoir");
    {
        parlay::internal::timer ht;
        ht.start();
        if (Hsize > 16) {
        std::cout << "Warning: at most 16 bits are supported for the hash size. "
                    << "Using 16 bits instead of " << Hsize << std::endl;
        Hsize = 16;
        }
        size_t root_fanout = static_cast<size_t>(fanout);
        if (!fanout_scheme.empty())
          root_fanout = std::stoull(fanout_scheme.substr(0, fanout_scheme.find(',')));
        root_fanout = std::min(root_fanout, size_t{512});
        pipnn_clean_configure_workspace(Points.dimension(), cluster_size, 512);
        const AEAMemoryPlan aea_memory_plan = make_aea_memory_plan(
            Points.size(), Points.dimension(), sizeof(typename Point::T),
            Points.aligned_bytes, parlay::num_workers(),
            64,
            7,
            cluster_size, 512, root_fanout, Hsize, prune_degree,
            0
            );
        if (std::getenv("PIPNN_AEA_PLAN_ONLY")) std::exit(0);
        SketchHash<Point, PointRange, indexType> H(Hsize, Points.dimension(),Points);
        ht.next("SketchHash construction time");
        ht.stop();
        //get size of H in bytes
        size_t h_size = H.get_total_size_in_bytes();
        std::cout << "Size of SketchHash (bytes): " << h_size << std::endl;

        UniformB64Graph<indexType, float,
            SketchHash<Point, PointRange, indexType>> lsh_graph(
                Points.size(), H, aea_memory_plan);
        build_index_impl(G, P, Points, cluster_rounds, cluster_size, MSTDeg, prune, prune_all, alpha, prune_degree, top_level_leaders, fanout, fanout_per_level, fraction_leaders, fanout_scheme, Hsize, table_size, lsh_graph, fixed_num_leaders);
    }
  }

  template<typename GraphH>
  void build_index_impl(GraphI &G, commandLine& P, PR &Points, long cluster_rounds,
                   long cluster_size, long MSTDeg, bool prune,
                   bool prune_all, double alpha,
                   long prune_degree,
                   long top_level_leaders, int fanout,
                   int fanout_per_level, double fraction_leaders,
                   std::string fanout_scheme, int Hsize, long table_size,
                   GraphH& graph_h, size_t fixed_num_leaders) {
    using BuildGraphH = GraphH;
    cluster<BuildGraphH, Point, PointRange, indexType> C;
    C.MST_DEG = MSTDeg;
    C.ALPHA = alpha;
    C.MAX_CLUSTER_SIZE = cluster_size;
    C.MAX_MERGED_CLUSTER_SIZE = cluster_size;
    C.MIN_CLUSTER_SIZE = cluster_size / 10;
    C.TOP_LEVEL_NUM_LEADERS = top_level_leaders;
    std::cout << "Set MSTDeg to: " << MSTDeg
              << " fanout: " << fanout
              << std::endl;
    C.FANOUT = fanout;
    C.FANOUT_PER_LEVEL = fanout_per_level;
    C.FRACTION_LEADERS = fraction_leaders;
    C.FIXED_NUM_LEADERS = fixed_num_leaders;
    C.LEVEL_BY_LEVEL = P.getOption("-levelbylevel");
    C.NO_MERGE = P.getOption("-nomerge");
    std::cout << "Sizeof indexType = " << sizeof(indexType) << std::endl;

    // parse fanout_scheme which will be something like "10,5,2,1" into a vector
    // of ints
    if (fanout_scheme != "") {
      std::stringstream ss(fanout_scheme);
      std::string token;
      while (std::getline(ss, token, ',')) {
        C.FANOUT_SCHEME.push_back(std::stoi(token));
      }
    }
    std::cout << "Fanout scheme: ";
    for (size_t i = 0; i < C.FANOUT_SCHEME.size(); i++) {
      std::cout << C.FANOUT_SCHEME[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "AEA_ONE_PASS mode="
              << (std::getenv("PIPNN_AEA_TIERS_IN") ? "oracle tiers" : "online")
              << '\n';
    C.multiple_clustertrees(graph_h, Points, cluster_size, cluster_rounds);


    if (std::getenv("PIPNN_STOP_AFTER_ACQUISITION")) {
      std::cout << "PIPNN_STOP_AFTER_ACQUISITION evidence_records=";
      uint64_t records = 0;
      for (size_t i = 0; i < Points.size(); ++i) records += graph_h.degree(i);
      std::cout << records << " avg="
                << (Points.size() ? double(records) / Points.size() : 0.0)
                << '\n';
      std::exit(0);
    }


    graph_h.release_hash_state();
    pipnn_clean_report_memory("after_candidate_generation_release");
    graph_h.report_budget_phase("after_candidate_generation_release");
    pipnn_clean_report_parlay_allocator("after_candidate_generation_release");
    std::cout << "pipnn_clean_census phase=after_candidate_generation_release"
              << " canonical_vector_bytes="
              << Points.size() * Points.aligned_bytes
              << " candidate_bytes=" << graph_h.candidate_live_bytes()
              << " final_graph_bytes=0\n";

    // Export the edges from the hashgraph into G.
    std::cout << "Allocating graph, prune_degree = " << prune_degree << std::endl;
#if (defined(PIPNN_LOWMEM_TWO_WITNESS) && defined(PIPNN_MEMORY_CLEAN)) || \
    defined(PIPNN_LOWMEM_EXACT_RR2W)
    if (
        table_size != 96
        || prune_degree !=
        lowmem_degree()
        ) {
      throw std::runtime_error(
          "LowMem layout has incompatible table_size or prune_degree");
    }
    std::cout << "Using exact in-place candidate-to-final conversion\n";
#else
    G.allocate_graph(prune_degree, Points.size());
#endif
    pipnn_clean_report_memory("after_final_graph_allocate");
    assert(G.max_degree() == prune_degree);
    ExportToGraph(graph_h, P, G, Points, prune_all, prune_degree, alpha);
    pipnn_clean_report_memory("after_final_prune");
    std::cout << "Average distance traversed: " << (dist_traversed.load() / Points.size()) << std::endl;
  }
};

}  // namespace parlayANN

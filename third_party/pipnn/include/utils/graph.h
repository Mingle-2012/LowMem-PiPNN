// This code is part of the Parlay Project
// Copyright (c) 2024 Guy Blelloch, Magdalen Dobson and the Parlay team
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

#include <algorithm>
#include <fcntl.h>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <bitset>
#include <atomic>

#include "../../../parlay/include/parlay/parallel.h"
#include "../../../parlay/include/parlay/primitives.h"
#include "../../../parlay/include/parlay/alloc.h"
#include "../../../parlay/include/parlay/internal/file_map.h"

#include "types.h"
#include "absl/synchronization/mutex.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"

#include <Eigen/Core>

namespace parlayANN {

template<typename indexType>
struct edgeRange {

  size_t size() const {return packed_degree ? *packed_degree : edges[0];}

  indexType id() const {return id_;}

  edgeRange() : edges(parlay::make_slice<indexType*, indexType*>(nullptr, nullptr)) {}

  edgeRange(indexType* start, indexType* end, indexType id, absl::Mutex* lock)
    : edges(parlay::make_slice<indexType*, indexType*>(start,end)), id_(id), lock(lock), packed_degree(nullptr) {
    maxDeg = edges.size() - 1;
  }

  // Compact graph rows keep degree out-of-line, so all row bytes are IDs.
  edgeRange(indexType* start, indexType* end, indexType* degree,
            indexType id, absl::Mutex* lock)
    : edges(parlay::make_slice<indexType*, indexType*>(start,end)), id_(id),
      lock(lock), packed_degree(degree) { maxDeg = edges.size(); }

  indexType operator [] (indexType j) const {
    if (j >= size()) {
      std::cout << "ERROR: index exceeds degree while accessing neighbors" << std::endl;
      abort();
    } else return edges[j + (packed_degree ? 0 : 1)];
  }

  // Not safe for concurrent updates.
  void append_neighbor(indexType nbh) {
    if (size() == maxDeg) {
      std::cout << "ERROR in append_neighbor for: " << id_ << ". Cannot exceed max degree "
                << maxDeg << std::endl;
      abort();
    } else {
      edges[size() + (packed_degree ? 0 : 1)] = nbh;
      set_size(size() + 1);
    }
  }

  template<typename rangeType>
  void update_neighbors(const rangeType& r){
    lock->Lock();
    if (r.size() > maxDeg) {
      std::cout << "ERROR in update_neighbors: cannot exceed max degree "
                << maxDeg << std::endl;
      abort();
    }
    set_size(r.size());
    for (int i = 0; i < r.size(); i++) {
      edges[i + (packed_degree ? 0 : 1)] = r[i];
    }
    lock->Unlock();
  }

  template<typename rangeType>
  void merge_neighbors(const rangeType& r) {
    lock->Lock();

    absl::flat_hash_set<indexType> neighbors;
    for (int i = 0; i < this->size(); ++i) {
      neighbors.insert((*this)[i]);
    }

    for (auto ngh : r) {
      neighbors.insert(ngh);
    }

    if (neighbors.size() > maxDeg) {
      std::cout << "ERROR in update_neighbors: cannot exceed max degree "
                << maxDeg << std::endl;
      abort();
    }
    set_size(neighbors.size());
    size_t i = 0;
    for (auto v : neighbors) {
      edges[i + (packed_degree ? 0 : 1)] = v;
      ++i;
    }
    lock->Unlock();
  }

  template<typename rangeType>
  void append_neighbors(const rangeType& r) {
    lock->Lock();
    if (r.size() + size() > maxDeg) {
      std::cout << "ERROR in append_neighbors for point " << id_
                << ": cannot exceed max degree " << maxDeg << std::endl;
      std::cout << size() << std::endl;
      std::cout << r.size() << std::endl;
      abort();
    }
    for (int i = 0; i < r.size(); i++) {
      edges[size() + i + (packed_degree ? 0 : 1)] = r[i];
    }
    set_size(size() + r.size());
    lock->Unlock();
  }

  void clear_neighbors(){
    lock->Lock();
    set_size(0);
    lock->Unlock();
  }

  void prefetch() const {
    int l = ((size() + (packed_degree ? 0 : 1)) * sizeof(indexType))/64;
    for (int i = 0; i < l; i++)
      __builtin_prefetch((char*) edges.begin() + i *  64);
  }

  template<typename F>
  void sort(F&& less){
    std::sort(begin(), end(), less);}

  void random_permute(){
    parlay::random_shuffle(parlay::slice<indexType*, indexType*>(begin(), end()), parlay::random(id_));
  }

  indexType* begin() const {return edges.begin() + (packed_degree ? 0 : 1);}

  indexType* end() const {return begin() + size();}

private:
  parlay::slice<indexType*, indexType*> edges;
  long maxDeg;
  indexType id_;
  absl::Mutex* lock;
  indexType* packed_degree = nullptr;
  void set_size(size_t n) { if (packed_degree) *packed_degree = n; else edges[0] = n; }
};

template<typename indexType_>
struct Graph {
  using indexType = indexType_;

  long max_degree() const {return maxDeg;}
  size_t size() const {return n;}

  Graph(){}

  void allocate_graph(long _maxDeg, size_t _n) {
    maxDeg = _maxDeg;
    n = _n;
    long cnt = n * (maxDeg + 1);
    long num_bytes = cnt * sizeof(indexType);
    indexType* ptr;
    if (num_bytes > 1 << 24) {
      constexpr size_t ALIGNMENT = 1L << 21;
      num_bytes = (num_bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
      ptr = (indexType*) aligned_alloc(ALIGNMENT, num_bytes);
      madvise(ptr, num_bytes, MADV_HUGEPAGE);
      graph = std::shared_ptr<indexType[]>(ptr, std::free);
    }
    else {
      ptr = (indexType*) parlay::p_malloc(num_bytes);
      graph = std::shared_ptr<indexType[]>(ptr, parlay::p_free);
    }
    // Every row is overwritten by prune_all before it becomes visible. Avoid
    // faulting unused adjacency tails and replace O(N) locks by striped locks.
    lock_count = std::min<size_t>(n, 1u << 16);
    locks = new absl::Mutex[lock_count];
  }

  // Adopt maxDeg tightly-packed IDs per row. Ownership of ids is transferred.
  void adopt_packed_graph(indexType* ids, indexType* degrees, long _maxDeg,
                          size_t _n) {
    maxDeg = _maxDeg; n = _n; packed = true;
    graph = std::shared_ptr<indexType[]>(ids, std::free);
    packed_degrees = std::shared_ptr<indexType[]>(degrees, std::free);
    lock_count = std::min<size_t>(n, 1u << 16);
    locks = new absl::Mutex[lock_count];
  }

  void adopt_mmap_packed_graph(indexType* ids, indexType* degrees,
                               long _maxDeg, size_t _n,
                               size_t mapped_bytes) {
    maxDeg = _maxDeg; n = _n; packed = true;
    graph = std::shared_ptr<indexType[]>(ids, [mapped_bytes](indexType* p) {
      if (p) ::munmap(p, mapped_bytes);
    });
    packed_degrees = std::shared_ptr<indexType[]>(degrees, std::free);
    lock_count = std::min<size_t>(n, 1u << 16);
    locks = new absl::Mutex[lock_count];
  }

  Graph(long maxDeg, size_t n) : maxDeg(maxDeg), n(n) {
    allocate_graph(maxDeg, n);
  }

  Graph(char* gFile){
    std::ifstream reader(gFile);
    if (!reader.is_open()) {
      std::cout << "graph file " << gFile << " not found" << std::endl;
      abort();
    }

    //read num points and max degree
    indexType num_points;
    indexType max_deg;
    reader.read((char*)(&num_points), sizeof(indexType));
    n = num_points;
    reader.read((char*)(&max_deg), sizeof(indexType));
    maxDeg = max_deg;
    std::cout << "Graph: detected " << num_points
              << " points with max degree " << max_deg << std::endl;

    //read degrees and perform scan to find offsets
    indexType* degrees_start = new indexType[n];
    reader.read((char*) (degrees_start), sizeof(indexType) * n);
    indexType* degrees_end = degrees_start + n;
    parlay::slice<indexType*, indexType*> degrees0 =
      parlay::make_slice(degrees_start, degrees_end);
    auto degrees = parlay::tabulate(degrees0.size(), [&] (size_t i){
      return static_cast<size_t>(degrees0[i]);});
    auto [o, total] = parlay::scan(degrees);
    auto offsets = o;
    std::cout << "Total edges read from file: " << total << std::endl;
    offsets.push_back(total);

    allocate_graph(max_deg, n);

    //write 1000000 vertices at a time
    size_t BLOCK_SIZE = 1000000;
    size_t index = 0;
    size_t total_size_read = 0;
    while(index < n){
      size_t g_floor = index;
      size_t g_ceiling = g_floor + BLOCK_SIZE <= n ? g_floor + BLOCK_SIZE : n;
      size_t total_size_to_read = offsets[g_ceiling] - offsets[g_floor];
      indexType* edges_start = new indexType[total_size_to_read];
      reader.read((char*) (edges_start), sizeof(indexType) * total_size_to_read);
      indexType* edges_end = edges_start + total_size_to_read;
      parlay::slice<indexType*, indexType*> edges =
        parlay::make_slice(edges_start, edges_end);
      indexType* gr = graph.get();
      parlay::parallel_for(g_floor, g_ceiling, [&] (size_t i){
        gr[i * (maxDeg + 1)] = degrees[i];
        for(size_t j = 0; j < degrees[i]; j++){
          gr[i * (maxDeg + 1) + 1 + j] = edges[offsets[i] - total_size_read + j];
        }
      });
      total_size_read += total_size_to_read;
      index = g_ceiling;
      delete[] edges_start;
    }
    delete[] degrees_start;
  }

  void save(char* oFile) {
    std::cout << "Writing graph with " << n
              << " points and max degree " << maxDeg
              << std::endl;
    parlay::sequence<indexType> preamble =
      {static_cast<indexType>(n), static_cast<indexType>(maxDeg)};
    parlay::sequence<indexType> sizes = parlay::tabulate(n, [&] (size_t i){
      return static_cast<indexType>((*this)[i].size());});
    std::ofstream writer;
    writer.open(oFile, std::ios::binary | std::ios::out);
    writer.write((char*) preamble.begin(), 2 * sizeof(indexType));
    writer.write((char*) sizes.begin(), sizes.size() * sizeof(indexType));
    size_t BLOCK_SIZE = 1000000;
    size_t index = 0;
    while(index < n){
      size_t floor = index;
      size_t ceiling = index + BLOCK_SIZE <= n ? index + BLOCK_SIZE : n;
      auto edge_data = parlay::tabulate(ceiling - floor, [&] (size_t i){
        return parlay::tabulate(sizes[i + floor], [&] (size_t j){
          return (*this)[i + floor][j];});
      });
      parlay::sequence<indexType> data = parlay::flatten(edge_data);
      writer.write((char*)data.begin(), data.size() * sizeof(indexType));
      index = ceiling;
    }
    writer.close();
  }

  edgeRange<indexType> operator [] (indexType i) {
    if (i > n) {
      std::cout << "ERROR: graph index out of range: " << i << std::endl;
      abort();
    }
    if (packed) return edgeRange<indexType>(graph.get() + i * maxDeg,
                                graph.get() + (i + 1) * maxDeg,
                                packed_degrees.get() + i, i,
                                locks + (i % lock_count));
    return edgeRange<indexType>(graph.get() + i * (maxDeg + 1),
                                graph.get() + (i + 1) * (maxDeg + 1),
                                i, locks + (i % lock_count));
  }

  ~Graph(){}

private:
  size_t n;
  long maxDeg;
  std::shared_ptr<indexType[]> graph;
  std::shared_ptr<indexType[]> packed_degrees;
  absl::Mutex* locks;
  size_t lock_count = 0;
  bool packed = false;
};

#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h" // Required for AbslHashValue
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"

// Forward declaration is good practice when a free function template depends on a class template
template<typename indexType_>
struct HashSetGraph;

// Used to aggregate per-point candidates in a hash table.
template<typename indexType_>
struct HashSetGraph {
public:
  using indexType = indexType_;
  void release_hash_state() {}
  void discard_candidate_rows(size_t, size_t) {}
  size_t candidate_live_bytes() const { return 0; }
#ifdef TW_REPRESENTATIVE_ORACLES
  size_t oracle_candidate_bytes(size_t) const { return 0; }
#endif
  struct Neighbor {
    indexType point_id;
    float norm;
#if defined(PIPNN_CONE_TRIE_FINALIZER) || \
    defined(PIPNN_DIRECTIONAL_EXTENSION_ORACLE)
    uint16_t hash = 0;
#endif

    // FIX 1: Add an equality operator.
    // The hash set uses this to check if two neighbors are the same.
    bool operator==(const Neighbor& other) const {
      return point_id == other.point_id;
    }

    template <typename H>
    friend H AbslHashValue(H h, const Neighbor& n) {
      return H::combine(std::move(h), n.point_id);
    }


  };
  using T = Neighbor;
  size_t size() const {return n;}

  HashSetGraph(){}

  void allocate_graph(size_t n) {
    locks = new absl::Mutex[n];
    tables = new absl::flat_hash_set<T>[n];
  }

  auto degree(size_t i) const {
    return tables[i].size();
  }

  HashSetGraph(size_t n) : n(n) {
    allocate_graph(n);
  }

  // FIX 3: Return a reference to the hash set. You cannot create a Span from it.
  auto& Neighbors(size_t i) {
    return tables[i];
  }
  template <class Range>
  void write_final_row(size_t, const Range&, size_t) {
    throw std::runtime_error("in-place handoff is unavailable with -no_reservoir");
  }
  void handoff_inplace_to(Graph<indexType>&, size_t) {
    throw std::runtime_error("in-place handoff is unavailable with -no_reservoir");
  }

  template<typename rangeType>
  void merge_neighbors(size_t i, const rangeType& r) {
    locks[i].Lock();
    // Assuming 'r' contains tuples, use structured bindings
    for (const auto& [point_id, norm] : r) {
      // Construct a Neighbor on the fly for insertion
      tables[i].insert({point_id, norm});
    }
    locks[i].Unlock();
  }

  void finalize() {}

  // Added a proper destructor to prevent memory leaks
  ~HashSetGraph(){
    if (locks) delete[] locks;
    if (tables) delete[] tables;
  }

  size_t n;
  absl::Mutex* locks = nullptr;
  absl::flat_hash_set<T>* tables = nullptr;
};


// Used to aggregate per-point candidates in a hash table.
// Also stores the floating point distance for each candidate.
// The neighbors are stored in sorted order.
template<typename indexType_>
struct NghDistGraphHeap {
public:
  using indexType = indexType_;
  using T = std::pair<indexType, float>;

  size_t size() const {return n;}
  static constexpr size_t kMaxDegree = 1024;

  static constexpr T kEmptyCell = std::make_pair(0, std::numeric_limits<float>::max());

  NghDistGraphHeap(){}

  void allocate_graph(size_t n) {
    locks = new absl::Mutex[n];
    constexpr size_t ALIGNMENT = 1L << 21;
    size_t num_bytes = ((kMaxDegree+1)*n*sizeof(T));
    tables = (T*) aligned_alloc(ALIGNMENT, num_bytes);
    madvise(tables, num_bytes, MADV_HUGEPAGE);
    std::cout << "Allocating: " << ((kMaxDegree+1)*n*sizeof(T)) << " bytes of space for the candidates lists." << std::endl;
    std::cout << "Sizeof T = " << sizeof(T) << std::endl;
    std::cout << tables[0].first << std::endl;
    std::cout << "Table size = " << tables << std::endl;
    parlay::internal::timer t;
    t.start();
    parlay::parallel_for(0, (kMaxDegree+1)*n, [&] (size_t i) {
      tables[i] = kEmptyCell;
    });
    t.next("Table initialization time");
  }

  NghDistGraphHeap(size_t n) : n(n) {
    allocate_graph(n);
  }

  size_t degree(size_t i) const {
    return tables[i*(kMaxDegree+1)].first;
  }

  absl::Span<T> Neighbors(size_t i) {
    size_t offset = i * (kMaxDegree + 1);
    size_t d = degree(i);
    return absl::Span<T>(tables + offset + 1, d);
  }

  absl::Span<const T> Neighbors(size_t i) const {
    size_t offset = i * (kMaxDegree + 1);
    size_t d = degree(i);
    return absl::Span<const T>(tables + offset + 1, d);
  }

  static bool dist_comparator(const T& a, const T& b) {
    return a.second < b.second;
  }

  template<typename rangeType>
  void merge_neighbors(size_t i, const rangeType& r) {
    locks[i].Lock();
    size_t offset = i * (kMaxDegree + 1);
    T* neighbors = tables + offset + 1;
    uint32_t& current_degree = tables[offset].first;

    // Interestingly, not doing this check and just returning false
    // (i.e., allowing dups) yields good quality on all datasets I
    // tested on while making the avg degree significantly lower.
    // Of course, if we run multiple clusters etc, performance starts
    // to tank, so avoiding the check is probably not very smart.
    auto is_present = [&] (const auto& ngh) {
      for (size_t i=0; i < current_degree; ++i) {
        if (ngh.first == neighbors[i].first) {
          return true;
        }
      }
      return false;
    };

    // r is sorted
    for (const auto& ngh : r) {
      if (current_degree < kMaxDegree && !is_present(ngh)) {
        neighbors[current_degree] = ngh;
        current_degree++;
        std::push_heap(neighbors, neighbors + current_degree, dist_comparator);
      } else {
        T& worst_neighbor = neighbors[0]; // Top of the max-heap
        if (ngh.second >= worst_neighbor.second) {
          break;
        }
        if (!is_present(ngh)) {
          std::pop_heap(neighbors, neighbors + kMaxDegree, dist_comparator);
          neighbors[kMaxDegree - 1] = ngh;
          std::push_heap(neighbors, neighbors + kMaxDegree, dist_comparator);
        }
      }
    }

    locks[i].Unlock();
  }

  void sort_heap(size_t i) {
    size_t offset = i * (kMaxDegree + 1);
    T* neighbors = tables + offset + 1;
    size_t current_degree = tables[offset].first;
    std::sort_heap(neighbors, neighbors + current_degree, dist_comparator);
  }

  void finalize() {
    parlay::parallel_for(0, n, [&] (size_t i) {
      sort_heap(i);
    });
  }

  ~NghDistGraphHeap(){}

  size_t n;
  absl::Mutex* locks;
  T* tables;
};


// Used to aggregate per-point candidates in a hash table.
// Also stores the floating point distance for each candidate.
// The neighbors are stored in sorted order.
template<typename indexType_>
struct NghDistGraph {
public:
  using indexType = indexType_;
  
  // CHANGED: Define a Neighbor struct to replace std::pair
  struct Neighbor {
    indexType point_id;
    float norm;
  };
  using T = Neighbor; // CHANGED: Use the new struct

  size_t size() const {return n;}
  static constexpr size_t kMaxDegree = 1024;

  NghDistGraph(){}

  void allocate_graph(size_t n) {
    locks = new absl::Mutex[n];
    constexpr size_t ALIGNMENT = 1L << 21;
    size_t num_bytes = ((kMaxDegree+1)*n*sizeof(T));
    tables = (T*) aligned_alloc(ALIGNMENT, num_bytes);
    madvise(tables, num_bytes, MADV_HUGEPAGE);
    std::cout << "Allocating: " << num_bytes << " bytes of space for the candidates lists." << std::endl;
    std::cout << "Sizeof T = " << sizeof(T) << std::endl;
    // CHANGED: Access .point_id instead of .first
    std::cout << tables[0].point_id << std::endl;
    std::cout << "Table size = " << tables << std::endl;
    parlay::parallel_for(0, n, [&] (size_t i) {
      // CHANGED: The degree is stored in the .point_id member
      tables[i*(kMaxDegree+1)].point_id = 0;
    });
    std::cout << "Cleared degree" << std::endl;
  }

  NghDistGraph(size_t n) : n(n) {
    allocate_graph(n);
  }

  size_t degree(size_t i) const {
    // CHANGED: Access .point_id for the degree
    return tables[i*(kMaxDegree+1)].point_id;
  }

  absl::Span<T> Neighbors(size_t i) {
    size_t offset = i * (kMaxDegree + 1);
    size_t d = degree(i);
    return absl::Span<T>(tables + offset + 1, d);
  }

  absl::Span<const T> Neighbors(size_t i) const {
    size_t offset = i * (kMaxDegree + 1);
    size_t d = degree(i);
    return absl::Span<const T>(tables + offset + 1, d);
  }

  template<typename rangeType>
  void merge_neighbors(size_t i, const rangeType& r) {
    locks[i].Lock();

    size_t offset = i * (kMaxDegree + 1);
    // CHANGED: The degree is stored in the .point_id member
    size_t current_degree = tables[offset].point_id;

    std::vector<T> merged_nghs;
    merged_nghs.reserve(kMaxDegree);

    auto it_existing = tables + offset + 1;
    auto end_existing = it_existing + current_degree;
    auto it_new = r.begin();
    auto end_new = r.end();

    while (merged_nghs.size() < kMaxDegree) {
        if (it_existing == end_existing && it_new == end_new) break;

        // CHANGED: Logic to handle merging Neighbor structs with incoming tuples
        bool take_from_existing;
        if (it_existing == end_existing) {
            take_from_existing = false; // Must take from new
        } else if (it_new == end_new) {
            take_from_existing = true; // Must take from existing
        } else {
            // Compare the norm of the existing Neighbor with the norm of the new tuple
            take_from_existing = (it_existing->norm <= std::get<1>(*it_new));
        }

        if (take_from_existing) {
            merged_nghs.push_back(*it_existing);
            // Check for and skip duplicates in the 'new' list
            if (it_new != end_new && it_existing->point_id == std::get<0>(*it_new)) {
              it_new++;
            }
            it_existing++;
        } else { // Take from new
            // Construct a Neighbor struct from the tuple
            const auto& [new_id, new_norm] = *it_new;
            merged_nghs.push_back({new_id, new_norm});
            if (new_id > n) {
              std::cerr << "Trying to add an id larger than n... " << new_id << std::endl;
              exit(0);
            }
            it_new++;
        }
    }

    // Copy merged neighbors back to the table
    for (size_t j = 0; j < merged_nghs.size(); ++j) {
      tables[offset + 1 + j] = merged_nghs[j];
      if (merged_nghs[j].point_id > n) {
        std::cerr << "Merging a neighbor with id = " << merged_nghs[j].point_id << " larger than n!" << std::endl;
        exit(-1);
      }
    }

    // Update the degree
    tables[offset].point_id = merged_nghs.size();

    locks[i].Unlock();
  }

  void finalize() { }

  // CHANGED: Added a destructor to prevent memory leaks
  ~NghDistGraph(){
    if (tables) free(tables);
    if (locks) delete[] locks;
  }

  size_t n;
  absl::Mutex* locks = nullptr;
  T* tables = nullptr;
};

template<typename indexType, typename distanceType>
struct LSHCandidate{
  indexType point_id;
  //distanceType norm;
  Eigen::bfloat16 norm;
  uint16_t hash;
};

template<typename indexType_, typename distanceType_, typename sketchHash>
struct LSHGraph {
public:
  using indexType = indexType_;
  using distanceType = distanceType_;
  using T = LSHCandidate<indexType,distanceType>;

  size_t size() const {return n;}

  // Projection scalars are not needed once streaming candidate generation has
  // finished.  Releasing them before final graph materialization is exact.
  void release_hash_state() { H.release_sketch_data(); }

  LSHGraph(){}

  void allocate_graph(size_t n) {
    locks = new absl::Mutex[n];
    constexpr size_t ALIGNMENT = 1L << 21;
    size_t num_bytes = ((kMaxDegree+1)*n*sizeof(T));
    tables = (T*) aligned_alloc(ALIGNMENT, num_bytes);
    madvise(tables, num_bytes, MADV_HUGEPAGE);
    std::cout << "Allocating: " << ((kMaxDegree+1)*n*sizeof(T)) << " bytes of space for the candidates lists." << std::endl;
    std::cout << "Sizeof T = " << sizeof(T) << std::endl;
    parlay::parallel_for(0, n, [&] (size_t i) {
      tables[i*(kMaxDegree+1)].point_id = 0;
      tables[i*(kMaxDegree+1)].norm = Eigen::bfloat16(0);
      tables[i*(kMaxDegree+1)].hash = 0;
    });
    std::cout << "Cleared degree" << std::endl;
  }

  LSHGraph(size_t n,size_t maxdeg, sketchHash &H, bool allocate=true) : n(n), H(H), kMaxDegree(maxdeg) {
    std::cout << "Allocate = " << allocate << std::endl;
    if (allocate) {
      std::cout << "allocating LSH graph" << std::endl;
      allocate_graph(n);
    }
  }

  void breakpoint() {
    volatile int a = 1;
    // std::cout << "Breakpoint hit!" << std::endl;
  }

  indexType &degree(size_t i) {
    return tables[i * (kMaxDegree + 1)].point_id;
  }

  Eigen::bfloat16 &get_furthest(size_t i) {
    return tables[i * (kMaxDegree + 1)].norm;
  }

  size_t get_furthest_index(size_t i) const {
    return static_cast<size_t>(tables[i * (kMaxDegree + 1)].hash);
  }

  T &get_candidate(size_t i, size_t j) {
    return tables[i * (kMaxDegree + 1) + 1 + j];
  }

  T *candidates_begin(size_t i) {
    return &tables[i * (kMaxDegree + 1) + 1];
  }

  T *candidates_end(size_t i) {
    return &tables[i * (kMaxDegree + 1) + 1 + degree(i)];
  }

  absl::Span<T> Neighbors(size_t i) {
    return absl::Span<T>(tables + i * (kMaxDegree + 1) + 1, degree(i));
  }

  absl::Span<const T> Neighbors(size_t i) const {
    return absl::Span<const T>(tables + i * (kMaxDegree + 1) + 1, degree(i));
  }

  void print_neighbors(size_t i) {
    std::cout << "Neighbors for point " << i << ": ";
    for (size_t j = 0; j < degree(i); ++j) {
      std::cout << "(" << get_candidate(i, j).point_id << ", "
                << get_candidate(i, j).norm << ", " << std::bitset<16>(get_candidate(i,j).hash) << ") ";
    }
    std::cout << std::endl;
  }

template<typename rangeType>
void merge_neighbors(const size_t i, const rangeType& r) {
  locks[i].Lock();

  const size_t base   = i * (kMaxDegree + 1);
  T* const sentinel   = &tables[base];        // degree in point_id; worst value in norm; worst index in hash
  T* const begin      = &tables[base + 1];

  size_t deg          = static_cast<size_t>(sentinel->point_id);
  T* end              = begin + deg;

  auto to_dt = [](Eigen::bfloat16 x) -> distanceType {
    return static_cast<distanceType>(static_cast<float>(x));
  };

  // Cached worst (kept in sync with sentinel)
  size_t       max_idx = static_cast<size_t>(sentinel->hash);
  distanceType max_val = to_dt(sentinel->norm);

  auto set_worst = [&](size_t idx, distanceType v) {
    max_idx = idx; max_val = v;
    sentinel->hash = static_cast<uint16_t>(idx);
    sentinel->norm = Eigen::bfloat16(v);
  };

  auto recompute_worst = [&]() {
    if (deg == 0) { set_worst(0, distanceType(0)); return; }
    size_t mi = 0;
    distanceType mv = to_dt(begin[0].norm);
    for (size_t j = 1; j < deg; ++j) {
      const distanceType v = to_dt(begin[j].norm);
      if (v > mv) { mv = v; mi = j; }
    }
    set_worst(mi, mv);
  };

  struct Cmp { bool operator()(const T& a, uint16_t b) const { return a.hash < b; } } cmp;

  for (const auto& cand : r) {
    const indexType    pid  = cand.first;
    const distanceType dist = cand.second;

    // Early break
    if (deg == kMaxDegree && dist >= max_val) break;

    const uint16_t cand_hash = static_cast<uint16_t>(H.relative_hash(i, pid));

    T* pos_ptr = std::lower_bound(begin, end, cand_hash, cmp);
    size_t pos = static_cast<size_t>(pos_ptr - begin);

    // Hash collision: keep nearer
    if (pos < deg && pos_ptr->hash == cand_hash) {
      const distanceType old = to_dt(pos_ptr->norm);
      if (old > dist) {
        pos_ptr->point_id = pid;
        pos_ptr->norm     = Eigen::bfloat16(dist);
        if (pos == max_idx) recompute_worst();
      }
      continue;
    }

    if (deg < kMaxDegree) {
      // make room
      std::move_backward(pos_ptr, end, end + 1);
      pos_ptr->point_id = pid;
      pos_ptr->norm     = Eigen::bfloat16(dist);
      pos_ptr->hash     = cand_hash;

      ++deg; ++end;

      // maintain worst
      if (deg == 1 || dist > max_val) {
        set_worst(pos, dist);
      } else if (pos <= max_idx) {
        set_worst(max_idx + 1, max_val);
      }
    } else {
      // full: replace worst while keeping hash order
      if (max_idx < pos) {
        // shift left [max_idx+1 .. pos-1] by 1
          std::move(begin + max_idx + 1, begin + pos, begin + max_idx);
        --pos;
      } else if (pos < max_idx) {
        // shift right [pos .. max_idx-1] by 1
        std::move_backward(begin + pos, begin + max_idx, begin + max_idx + 1);
      }
      // overwrite
      T* write = begin + pos;
      write->point_id = pid;
      write->norm     = Eigen::bfloat16(dist);
      write->hash     = cand_hash;

      recompute_worst();
    }
  }

  sentinel->point_id = static_cast<indexType>(deg);
  locks[i].Unlock();
}

//   template<typename rangeType>
// void merge_neighbors(const size_t i, const rangeType& r) {
//   // RAII lock is safer; if you prefer raw Lock/Unlock for micro reasons, keep it.
//   locks[i].Lock();

//   // Local, cached pointers/degree.
//   T* const begin = &tables[i * (kMaxDegree + 1) + 1];
//   uint16_t deg   = static_cast<uint16_t>(tables[i * (kMaxDegree + 1)].point_id);
//   T*       end   = begin + deg;

//   // One-time scan for current worst; maintain incrementally afterwards.
//   uint16_t max_idx = 0;
//   distanceType max_val = distanceType( (deg > 0) ? begin[0].norm : 0 );
//   for (uint16_t j = 1; j < deg; ++j) {
//     const distanceType v = begin[j].norm;
//     if (v > max_val) { max_val = v; max_idx = j; }
//   }

//   // Tiny functor to help inlining of lower_bound
//   struct Cmp {
//     bool operator()(const T& a, uint16_t b) const { return a.hash < b; }
//   } cmp;

//   for (const auto& cand : r) { // no copies
//     const indexType     pid  = cand.first;
//     const distanceType  dist = cand.second;

//     // If full and candidate is not better than worst, we can stop (r is sorted by dist).
//     if (deg == kMaxDegree && dist >= max_val) {
//       break; // [[likely]]
//     }

//     // Hash once
//     const uint16_t cand_hash = static_cast<uint16_t>(H.relative_hash(i, pid));

//     // Binary search on hash
//     T* const pos_ptr = std::lower_bound(begin, end, cand_hash, cmp);
//     uint16_t pos = static_cast<uint16_t>(pos_ptr - begin);

//     // Collision: keep nearer
//     if (pos < deg && pos_ptr->hash == cand_hash) {
//       if (pos_ptr->norm > dist) {
//         pos_ptr->point_id = pid;
//         pos_ptr->norm     = Eigen::bfloat16(dist);
//         // If we just improved the previous max slot, recompute max lazily.
//         if (pos == max_idx) {
//           // Recompute worst (deg ≤ 192, cheap and rare)
//           max_idx = 0; max_val = begin[0].norm;
//           for (uint16_t j = 1; j < deg; ++j) {
//             const distanceType v = begin[j].norm;
//             if (v > max_val) { max_val = v; max_idx = j; }
//           }
//         } 
//       }
//       continue;
//     }

//     if (deg < kMaxDegree) {
//       // INSERT (make room at pos)
//       //if constexpr (std::is_trivially_copyable_v<T>) {
//       //  std::memmove(pos_ptr + 1, pos_ptr, (end - pos_ptr) * sizeof(T));
//       //} else {
//         std::move_backward(pos_ptr, end, end + 1);
//       //}
//       pos_ptr->point_id = pid;
//       pos_ptr->norm     = Eigen::bfloat16(dist);
//       pos_ptr->hash     = cand_hash;
//       ++deg; ++end;

//       // Maintain worst incrementally
//       if (dist > max_val) { max_val = dist; max_idx = pos; }
//       else if (max_idx >= pos) { ++max_idx; /* element shifted right */ }
//     } else {
//       // FULL: replace worst by inserting at correct hash position with one splice.
//       // We remove begin[max_idx] and insert new at 'pos' to keep sorted-by-hash.
//       // Compute the gap shift with a single memmove.
//       if (max_idx < pos) {
//         // We shift left the range (max_idx+1 .. pos-1) by one.
//         // if constexpr (std::is_trivially_copyable_v<T>) {
//         //   std::memmove(begin + max_idx, begin + max_idx + 1, (pos - max_idx - 1) * sizeof(T));
//         // } else {
//           std::move(begin + max_idx + 1, begin + pos, begin + max_idx);
//         //}
//         pos -= 1; // insertion index changes after removing at max_idx
//       } else if (pos < max_idx) {
//         // Shift right the range (pos .. max_idx-1) by one.
//         // if constexpr (std::is_trivially_copyable_v<T>) {
//         //   std::memmove(begin + pos + 1, begin + pos, (max_idx - pos) * sizeof(T));
//         // } else {
//           std::move_backward(begin + pos, begin + max_idx, begin + max_idx + 1);
//       //  }
//       } else {
//         // pos == max_idx: direct overwrite is fine; no shifting needed.
//       }

//       // Insert in-place
//       T* write = begin + pos;
//       write->point_id = pid;
//       write->norm     = Eigen::bfloat16(dist);
//       write->hash     = cand_hash;

//       // Recompute worst lazily only if we overwrote it or likely changed it.
//       // (Cheap for deg ≤ 192; avoids subtle incremental bookkeeping.)
//       max_idx = 0; max_val = begin[0].norm;
//       for (uint16_t j = 1; j < deg; ++j) {
//         const distanceType v = begin[j].norm;
//         if (v > max_val) { max_val = v; max_idx = j; }
//       }
//     }
//   }

//   // Write back degree once.
//   tables[i * (kMaxDegree + 1)].point_id = static_cast<indexType>(deg);

//   locks[i].Unlock();
// }



  void finalize() {}

  ~LSHGraph(){}

  size_t n;
  absl::Mutex* locks;
  T* tables;
  sketchHash &H;
  size_t kMaxDegree;
};



} // end namespace

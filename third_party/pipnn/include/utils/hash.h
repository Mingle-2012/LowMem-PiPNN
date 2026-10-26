#include "../../../parlay/include/parlay/parallel.h"
#include "../../../parlay/include/parlay/primitives.h"
#include "../../../parlay/include/parlay/random.h"
#include <immintrin.h>

#include <cmath>  // For sqrt
#include <iomanip> // For formatting output

#pragma once


template<typename Point, typename PointRange, typename indexType>
struct Hash_Family{
public:
    Hash_Family(indexType num_hashes, indexType dim, int seed = 0)
        : num_hashes(num_hashes), dim(dim), gen(parlay::random_generator(seed)) {
        for (auto i = 0; i < num_hashes; i++) {
            // normal_distribution caches state: sharing it across tabulate
            // workers is a data race. Fixed-stream owner controls need the
            // same immutable directions, independent of task scheduling.
            parlay::sequence<float> vec(dim);
            for(size_t j=0;j<dim;++j){auto r=gen[(i+1)*dim+j];vec[j]=normal_dist(r);}
            random_vectors.push_back(vec);  
        }
    }

    ~Hash_Family() = default;

    void add_vectors(int num) {
        num_hashes += num;
        for (auto i = 0; i < num; i++) {
            parlay::sequence<float> vec(dim);
            for(size_t j=0;j<dim;++j){auto r=gen[(num_hashes+i+1)*dim+j];vec[j]=normal_dist(r);}
            random_vectors.push_back(vec);  
        }
    }

    // accessor functions
    indexType get_num_hashes() const {
        return num_hashes;
    }
    indexType get_dim() const {
        return dim;
    }
    const parlay::sequence<float>& get_random_vector(indexType i) const {
        return random_vectors[i];
    }

    size_t get_random_vectors_size_in_bytes() const {
        size_t total_size = 0;
        if (!random_vectors.empty()) {
            // Assumes all random vectors have the same size (dim)
            total_size = num_hashes * dim * sizeof(float);
        }
        return total_size;
    }
    
    
private:
    indexType num_hashes;
    indexType dim;
    parlay::random_generator gen;
    std::vector<parlay::sequence<float>> random_vectors;
    std::normal_distribution<float> normal_dist{0.0f, 1.0f};
};

template<typename Point, typename PointRange, typename indexType>
struct SimHash{
public:
    using distanceType = typename Point::distanceType;

    SimHash(indexType num_hashes, indexType dim, int seed = 0)
        : hash_family(num_hashes, dim, seed) {}

    //constructor that takes a hash_family
    SimHash(Hash_Family<Point, PointRange, indexType> &hash_family)
        : hash_family(hash_family) {}


    ~SimHash() = default;

    parlay::sequence<std::pair<size_t,indexType>> generate_hash_values(PointRange& points, parlay::sequence<indexType>& active_indices) {
        size_t num_vectors = hash_family.get_num_hashes();
        size_t d = hash_family.get_dim();
        
        parlay::sequence<std::pair<size_t,indexType>> hash_values = parlay::tabulate(active_indices.size(), [&](size_t i) {
            size_t hash = 0;
            for (size_t j = 0; j < num_vectors; j++) {
                float dot = 0;
                for (size_t k = 0; k < d; k++) {
                    dot += points[active_indices[i]][k] * hash_family.get_random_vector(j)[k];
                }
                hash = hash << 1;
                if (dot > 0) {
                    hash |= 1;
                }
            }
            return std::make_pair(hash, active_indices[i]);
        });
        return hash_values;
    }

    parlay::sequence<std::pair<size_t,indexType>> generate_relative_hash_values(PointRange& points, parlay::sequence<indexType>& active_indices, indexType source_idx) {
        size_t num_vectors = hash_family.get_num_hashes();
        size_t d = hash_family.get_dim();
        
        parlay::sequence<std::pair<size_t,indexType>> hash_values = parlay::tabulate(active_indices.size(), [&](size_t i) {
            size_t hash = 0;
            for (size_t j = 0; j < num_vectors; j++) {
                float dot = 0;
                for (size_t k = 0; k < d; k++) {
                    dot += (points[active_indices[i]][k] - points[source_idx][k]) * hash_family.get_random_vector(j)[k];
                }
                hash = hash << 1;
                if (dot > 0) {
                    hash |= 1;
                }
            }
            return std::make_pair(hash, active_indices[i]);
        });
        return hash_values;
    }

    uint32_t relative_hash(PointRange& points, indexType p, indexType q) {
        uint32_t hash = 0;
        auto num_hashes = hash_family.get_num_hashes();
        auto d = hash_family.get_dim();
        for (size_t i = 0; i < num_hashes; ++i) {
            float dot = 0;
            for (size_t j = 0; j < d; ++j) {
                dot += (points[q][j] - points[p][j]) * hash_family.get_random_vector(i)[j];
            }
            hash = hash << 1;
            if (dot > 0) {
                hash |= 1;
            }
        }
        return hash;
    }


    //hash against a single hash vector
    size_t hash_single_vector(const Point& point, indexType vector_index) const {
        const auto& random_vector = hash_family.get_random_vector(vector_index);
        float dot = 0;
        for (size_t k = 0; k < hash_family.get_dim(); k++) {
            dot += point[k] * random_vector[k];
        }
        return (dot > 0) ? 1 : 0;
    }


private:
    Hash_Family<Point, PointRange, indexType> hash_family;
};

template<typename Point, typename PointRange, typename indexType>
struct SketchHash{
public:
    using distanceType = typename Point::distanceType;
    using PR = PointRange;

    SketchHash(indexType num_hashes, indexType dim, int seed = 0)
        : hash_family(num_hashes, dim, seed) {}

    ~SketchHash() = default;

    SketchHash(indexType num_hashes, indexType dim, PR &Points, int seed = 0)
        : hash_family(num_hashes, dim, seed) {
            n = Points.size();
            size_t num_bytes = num_hashes * sizeof(distanceType);
            size_t total_bytes = num_bytes * n;
            constexpr size_t ALIGNMENT = 1ULL << 21;
            total_bytes = (total_bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
            distanceType* ptr = (distanceType*) aligned_alloc(ALIGNMENT, total_bytes);
            madvise(ptr, total_bytes, MADV_HUGEPAGE);
            sketch_data = std::shared_ptr<distanceType[]>(ptr, std::free);
            parlay::parallel_for(0,n,[&] (size_t i) {
                for (auto j = 0; j < num_hashes; ++j) {
                    sketch_data[(i*num_hashes)+j] = sketch(Points[i],j);
                }
            });
        }

    SketchHash(Hash_Family<Point, PointRange, indexType> &hash_family, PR &Points)
        : hash_family(hash_family) {
            n = Points.size();
            size_t num_bytes = hash_family.get_num_hashes() * sizeof(distanceType);
            size_t total_bytes = num_bytes * n;
            constexpr size_t ALIGNMENT = 1ULL << 21;
            total_bytes = (total_bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
            distanceType* ptr = (distanceType*) aligned_alloc(ALIGNMENT, total_bytes);
            madvise(ptr, total_bytes, MADV_HUGEPAGE);
            sketch_data = std::shared_ptr<distanceType[]>(ptr, std::free);
            parlay::parallel_for(0,n,[&] (size_t i) {
                for (auto j = 0; j < hash_family.get_num_hashes(); ++j) {
                    sketch_data[(i*hash_family.get_num_hashes())+j] = sketch(Points[i],j);
                }
            });
        }

    distanceType sketch(Point &&point, indexType vector_index) {
        auto& random_vector = hash_family.get_random_vector(vector_index);
        distanceType dot = 0;
        for (size_t k = 0; k < hash_family.get_dim(); k++) {
            dot += point[k] * random_vector[k];
        }
        return dot;
    }

    uint32_t relative_hash(size_t p, size_t q) {
        uint32_t hash=0;
        auto num_hashes = hash_family.get_num_hashes();
        auto diff = sketch_data[q*num_hashes] - sketch_data[p*num_hashes];
        hash |= (diff > 0);
        for (auto dim = 1; dim < num_hashes; ++dim) {
            diff =  sketch_data[(q*num_hashes)+dim] - sketch_data[(p*num_hashes)+dim];
            hash = hash << 1;
            hash |= (diff > 0);
        }
        return hash;
    }

    size_t get_total_size_in_bytes() const {
        size_t sketch_data_size = n * hash_family.get_num_hashes() * sizeof(distanceType);
        size_t random_vectors_size = hash_family.get_random_vectors_size_in_bytes();
        return sketch_data_size + random_vectors_size;
    }

    // Candidate hashes are no longer needed after streaming reduction.  The
    // isolated low-memory builder calls this before final graph materialization.
    void release_sketch_data() { sketch_data.reset(); }


private:
    Hash_Family<Point, PointRange, indexType> hash_family;
    std::shared_ptr<distanceType[]> sketch_data;
    long aligned_bytes;
    size_t n;

};

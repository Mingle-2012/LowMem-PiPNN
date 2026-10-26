#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
namespace orderleaf {
struct __attribute__((packed)) Edge {uint32_t id;uint16_t norm;};
static_assert(sizeof(Edge)==6,"ordered intermediate edge layout");
void build(const float* data,size_t n,size_t d,size_t pitch,
 const std::vector<uint64_t>& leaf_offsets,std::vector<uint32_t>& leaf_ids,
 const std::vector<uint64_t>& source_offsets,const std::vector<uint32_t>& source_leaves,
 Edge* output,uint8_t* degrees,double alpha,bool use_bounds,bool check);
void unit();
}

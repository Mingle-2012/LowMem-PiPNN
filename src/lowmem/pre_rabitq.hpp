#pragma once
#include <cstddef>
#include <cstdint>
namespace brq {
void configure(const float* raw,size_t n,size_t d);
void prepare(size_t threads);
void query(size_t u);
float estimate(uint32_t v);
void release();
void release_after_topology();
bool live();
void unit();
}

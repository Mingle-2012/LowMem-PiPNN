#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
struct B64Seed { std::vector<size_t> offsets; std::vector<uint32_t> neighbors;
};
B64Seed build_b64_resident(const float*,size_t,size_t,bool,double);

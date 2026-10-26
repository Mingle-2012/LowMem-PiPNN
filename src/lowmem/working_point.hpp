// SPDX-License-Identifier: MIT
#pragma once
#include <cstdlib>
#include <string>
#include <stdexcept>
inline constexpr size_t lowmem_degree(){return 32;}
inline constexpr size_t lowmem_build_ef(){return 400;}
inline size_t lowmem_threads(){static const size_t t=[](){const char* e=std::getenv("LOWMEM_THREADS");size_t n=e?std::stoul(e):128;if(!n||n>128)throw std::runtime_error("threads must be in1..128");return n;}();return t;}

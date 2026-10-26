// SPDX-License-Identifier: MIT
#pragma once
#include <lowmem_pipnn/index.hpp>
#include <rabitqlib/utils/int8_l2.hpp>
#include "profile.hpp"
#include <iostream>
#include <unordered_set>
#include <random>
#include <chrono>
#include <iomanip>
namespace rabitqlib::symqg {using CandidateList=std::vector<AnnCandidate<float>>;inline constexpr size_t kMaxBsIter=5;}

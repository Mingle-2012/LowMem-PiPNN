// SPDX-License-Identifier: Apache-2.0
// Extracted QuantizedQuery helper from RaBitQ-Library's SymphonyQG implementation.
#pragma once
#include <numeric>
#include "rabitqlib/utils/space.hpp"
namespace rabitqlib::symqg {
class QuantizedQuery {
 const float* rotated_query_;float k1xsumq_,g_add_;
 public:
 QuantizedQuery(const float* q,const float* centroid,size_t p,MetricType metric):rotated_query_(q){k1xsumq_=std::accumulate(q,q+p,0.F)/-2;g_add_=metric==METRIC_IP?-dot_product(q,centroid,p):euclidean_sqr(q,centroid,p);}
 const float* rotated_query()const{return rotated_query_;}float k1xsumq()const{return k1xsumq_;}float g_add()const{return g_add_;}
};
}

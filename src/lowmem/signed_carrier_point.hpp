#pragma once
#include "../../third_party/rabitq/include/rabitqlib/utils/int8_l2.hpp"
#include "../../third_party/pipnn/include/utils/euclidian_point.h"
// SpaceV has a single owning FP32 matrix required by RAW-QG. Signed bytes are
// exactly representable there. This point view performs INTEGER subtraction,
// squares and accumulation via the existing SIMD int8 carrier kernel; it never
// creates an N*D byte matrix or substitutes FP32 GEMM for integer routing/leaf.
struct SignedCarrierPoint : parlayANN::Euclidian_Point<float> {
 using Base=parlayANN::Euclidian_Point<float>;
 static constexpr bool signed_byte_carrier=true;
 SignedCarrierPoint()=default;
 SignedCarrierPoint(uint8_t* values,long id,parameters params):Base(values,id,params),dims_(params.dims){}
 float distance(const SignedCarrierPoint& other)const{return rabitqlib::int8_l2_carrier(data(),other.data(),dims_);}
 private:int dims_=0;
};

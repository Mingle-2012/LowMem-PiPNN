#include "pre_rabitq.hpp"
#include <rabitqlib/utils/rotator.hpp>
#include <rabitqlib/index/quantized_query.hpp>

#include "../../third_party/rabitq/include/rabitqlib/quantization/rabitq.hpp"
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <limits>
namespace brq {
using namespace rabitqlib;
namespace {
const float* raw=nullptr;size_t n=0,d=0,padded=0,stride=0;
std::unique_ptr<char[]> cache;
std::unique_ptr<Rotator<float>> rotator;
std::vector<float> center;
ex_ipfunc kernel=nullptr;
thread_local std::vector<float> rotated;
// Read the out-of-line query getters once per source, not twice per candidate.
// Keep k1xsumq unscaled so full_est_dist retains its original FP32/FMA ordering.
thread_local float query_add=0,query_sum=0;
constexpr size_t bits=4;
constexpr MetricType metric=METRIC_L2;
}
void configure(const float* data,size_t count,size_t dim){if(cache)throw std::runtime_error("pre RaBitQ already live");raw=data;n=count;d=dim;}
void prepare(size_t threads){
 auto start=std::chrono::steady_clock::now();
 if(!raw||!n||!d)throw std::runtime_error("missing pre RaBitQ input");
 padded=((d+63)/64)*64;stride=ExDataMap<float>::data_bytes(padded,bits);
 rotator.reset(choose_rotator<float>(d,RotatorType::FhtKacRotator,padded,12345));
 // Bounded deterministic partial sums, no N*d temporary matrix.
 constexpr size_t chunk=8192;size_t blocks=(n+chunk-1)/chunk;
 std::vector<float> mean(d,0);
 {
 {
 std::vector<double> partial(blocks*d,0);
#pragma omp parallel for num_threads(threads) schedule(static)
 for(size_t b=0;b<blocks;++b)for(size_t i=b*chunk;i<std::min(n,(b+1)*chunk);++i)for(size_t k=0;k<d;++k)partial[b*d+k]+=raw[i*d+k];
 for(size_t k=0;k<d;++k){double sum=0;for(size_t b=0;b<blocks;++b)sum+=partial[b*d+k];mean[k]=sum/n;}
 }
 } // Release centroid partials before allocating the quantization cache.
 center.resize(padded);rotator->rotate(mean.data(),center.data());
 cache.reset(new char[n*stride]);kernel=select_excode_ipfunc(bits);
 auto config=quant::faster_config(padded,bits);
#pragma omp parallel num_threads(threads)
 {
  std::vector<float> scratch(padded);std::vector<uint8_t> unpacked(padded);
#pragma omp for schedule(static)
  for(size_t i=0;i<n;++i){
   rotator->rotate(raw+i*d,scratch.data());ExDataMap<float> out(cache.get()+i*stride,padded,bits);
   float add,scale,error;
   quant::quantize_full_single(scratch.data(),center.data(),padded,bits,unpacked.data(),add,scale,error,metric,config);
   out.f_add_ex()=add;out.f_rescale_ex()=scale;
   quant::rabitq_impl::ex_bits::packing_rabitqplus_code(unpacked.data(),out.ex_code(),padded,bits);
  }
 }
 std::cout<<std::setprecision(12)<<"PRE_RABITQ_PREP bits="<<bits<<" padded_dim="<<padded<<" seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<" cache_payload_bytes="<<n*stride<<" rotated_global_bytes=0 final_layout_built=0 build_threads="<<threads<<" centroid="<<(metric==METRIC_IP?"zero":"global_mean")<<" metric="<<(metric==METRIC_IP?"IP":"L2")<<" config=faster_config seed=12345"<<std::endl;
}
bool live(){return bool(cache);}
void query(size_t u){if(!cache||!raw||u>=n)throw std::runtime_error("pre RaBitQ query requires live cache/valid source");rotated.resize(padded);rotator->rotate(raw+u*d,rotated.data());symqg::QuantizedQuery prepared(rotated.data(),center.data(),padded,metric);query_add=prepared.g_add();query_sum=prepared.k1xsumq();
}
float estimate(uint32_t v){ConstExDataMap<float> point(cache.get()+size_t(v)*stride,padded,bits);return quant::full_est_dist(point.ex_code(),rotated.data(),kernel,padded,bits,point.f_add_ex(),point.f_rescale_ex(),query_add,query_sum);}
void release(){cache.reset();rotator.reset();std::vector<float>().swap(center);raw=nullptr;std::cout<<"PRE_RABITQ_RELEASE cache_live_bytes=0 before_downstream_topology=1"<<std::endl;}
void release_after_topology(){cache.reset();rotator.reset();std::vector<float>().swap(center);raw=nullptr;std::cout<<"PRE_RABITQ_RELEASE cache_live_bytes=0 before_downstream_topology=0 after_downstream_topology=1 before_final_encoding=1"<<std::endl;}
void unit(){
 std::mt19937 gen(12345);std::normal_distribution<float> sample;
 for(size_t dim:{size_t(96),size_t(100),size_t(128),size_t(768)})for(bool zero:{false,true}){
  size_t count=96;std::vector<float> data(count*dim);for(auto& x:data)x=zero?0.f:sample(gen);configure(data.data(),count,dim);prepare(4);
  for(size_t u=0;u<count;u+=7){query(u);symqg::QuantizedQuery reference(rotated.data(),center.data(),padded,metric);for(size_t v=0;v<count;++v){
   ConstExDataMap<float> point(cache.get()+v*stride,padded,bits);double inner=0;
   for(size_t i=0;i<padded;i+=16)for(size_t j=0;j<8;++j){uint8_t pair=point.ex_code()[i/2+j];inner+=double(rotated[i+j])*(pair&15)+double(rotated[i+j+8])*(pair>>4);}
   double expected=point.f_add_ex()+query_add+double(point.f_rescale_ex())*(inner+double(query_sum)*15);
   float actual=estimate(v);if(!std::isfinite(actual)||std::abs(actual-expected)>2e-4*(1+std::abs(expected)))throw std::runtime_error("RaBitQ SIMD/scalar decoded distance mismatch");
   if(metric==METRIC_IP && (point.f_add_ex()!=1.F || query_add!=0.F || point.f_rescale_ex()>0.F))throw std::runtime_error("IP zero-center direct factors mismatch");
   float original=quant::full_est_dist(point.ex_code(),rotated.data(),kernel,padded,bits,point.f_add_ex(),point.f_rescale_ex(),reference.g_add(),reference.k1xsumq());
   if(std::memcmp(&actual,&original,sizeof(float)))throw std::runtime_error("RaBitQ cached query factors FP32 mismatch");
  }}
  release();
 }
 std::cout<<"PRE_RABITQ_UNIT dims=96,100,128,768 metric="<<(metric==METRIC_IP?"IP":"L2")<<" zero_residual=passed native_packing_scalar_estimator=passed cached_query_factors_fp32=passed"<<std::endl;
}
}

#include "capture.hpp"
#include "working_point.hpp"
#include "profile.hpp"
#include <cstring>
#include <memory>
#include "../../third_party/pipnn/include/utils/euclidian_point.h"

#include "../../third_party/pipnn/include/utils/point_range.h"
#include "../../third_party/pipnn/include/utils/edge_productivity.h"
#ifdef B_UNIQUE_SIGNED_CARRIER
#include "signed_carrier_point.hpp"
#endif
static thread_local B64Seed* evidence_sink=nullptr;
template<class G> void handoff_capture(G& graph,size_t n){
 if(!evidence_sink)return;
 auto& out=*evidence_sink;out.offsets.resize(n+1);
 for(size_t u=0;u<n;++u)out.offsets[u+1]=out.offsets[u]+graph.degree(u);
 out.neighbors.resize(out.offsets.back());
 parlay::parallel_for(0,n,[&](size_t u){size_t k=out.offsets[u];for(auto e:graph.Neighbors(u))out.neighbors[k++]=e.point_id;});
}
#include "pre_rabitq.hpp"
#include "pre_stage.hpp"
template<class G> void handoff_pre_prepare(G&,size_t){if(bpre::mode){if(!evidence_sink)throw std::runtime_error("pre stage requires immutable evidence capture");
 bpre::prepare(*evidence_sink);
 brq::prepare(parlay::num_workers());
 }}
#include "pipnn.hpp"

// Same Uniform-B64 implementation/configuration as the existing native builder.
// All file loading/graph serialization is bypassed. Memory layout conversion and
// CSR handoff belong to the end-to-end interval, not preprocessing outside it.
template<class T> static B64Seed build(const float* raw,size_t n,size_t d,double alpha){
#ifdef B_UNIQUE_SIGNED_CARRIER
    using Point=std::conditional_t<std::is_same_v<T,float>,SignedCarrierPoint,parlayANN::Euclidian_Point<T>>;
#else
    using Point=parlayANN::Euclidian_Point<T>;
#endif
    using PR=parlayANN::PointRange<Point>;
    lowmem_profile::mark("B64_layout_copy");
    PR points;points.params=typename Point::parameters(d);points.n=n;
    points.aligned_bytes=((d*sizeof(T)+63)/64)*64;
    size_t bytes=((n*points.aligned_bytes+(1<<21)-1)/(1<<21))*(1<<21);
    bool borrow=false;
    borrow=std::getenv("LOWMEM_UNIQUE_VECTOR")&&std::atoi(std::getenv("LOWMEM_UNIQUE_VECTOR"));
    if(borrow&&(!std::is_same_v<T,float>||reinterpret_cast<uintptr_t>(raw)%64))throw std::runtime_error("unique vector requires FP32 and aligned source");
    // Native point distances use scalar/unaligned-safe accesses, and leaf GEMM
    // copies rows to its existing workspace. PointRange supports byte stride.
    if(borrow)points.aligned_bytes=d*sizeof(float);
    auto* memory=borrow?reinterpret_cast<uint8_t*>(const_cast<float*>(raw)):static_cast<uint8_t*>(aligned_alloc(1<<21,bytes));
    if(!memory)throw std::bad_alloc();
    points.values=borrow?std::shared_ptr<uint8_t[]>(memory,[](uint8_t*){}):std::shared_ptr<uint8_t[]>(memory,std::free);
    if(!borrow)parlay::parallel_for(0,n,[&](size_t i){
        auto* dst=reinterpret_cast<T*>(points.location(i));
        for(size_t j=0;j<d;++j)dst[j]=static_cast<T>(raw[i*d+j]);
    });
    std::cout<<"UNIQUE_VECTOR enabled="<<borrow<<" aligned_copy_bytes="<<(borrow?0:bytes)<<" source_bytes="<<n*d*sizeof(float)<<" FP32_unchanged=1 row_stride="<<points.aligned_bytes<<std::endl;
#ifdef B_UNIQUE_SIGNED_CARRIER
    if constexpr(std::is_same_v<T,float>)std::cout<<"SIGNED_CARRIER source_matrices=1 global_int8_copy_bytes=0 routing=integer_SIMD leaf=integer_SIMD pruning=integer_SIMD signed_range=-128:127 FP32_storage_lossless=1"<<std::endl;
#endif
    std::vector<std::string> args={"resident-b64","-multi_pivot","-prune","-prune_all","-leaf_method","EigenKNN"};
    std::vector<char*> argv;for(auto& s:args)argv.push_back(s.data());
    commandLine options(argv.size(),argv.data());
    parlayANN::Graph<uint32_t> graph;
    parlayANN::pipnn_index<Point,PR,uint32_t> index;
    lowmem_profile::mark("B64_build");
    constexpr size_t leaf_k=2;
#ifdef LOWMEM_INT8_L2
    // Rotation/quantization uses FP32, but decoded signed bytes are exact in
    // this carrier. Selected distances and initial prune still use Point<int8_t>.
    brq::configure(raw,n,d);
#else
    if constexpr(std::is_same_v<T,float>)brq::configure(raw,n,d);
    else throw std::runtime_error("pre RaBitQ variant requires FP32 L2 input");
#endif
    index.build_index(graph,options,points,1,1024,leaf_k,true,true,alpha,lowmem_degree(),1000,30,0,.005,"10,3",12,96,false,0);
    if(bpre::mode)bpre::finish();
    lowmem_profile::mark("CSR_handoff");
    B64Seed out;out.offsets.resize(n+1);
    for(size_t u=0;u<n;++u)out.offsets[u+1]=out.offsets[u]+graph[u].size();
    out.neighbors.resize(out.offsets.back());
    parlay::parallel_for(0,n,[&](size_t u){auto row=graph[u];for(size_t k=0;k<row.size();++k)out.neighbors[out.offsets[u]+k]=row[k];});
    return out;
}
B64Seed captured_b64(const float* raw,size_t n,size_t d,bool signed_bytes,double alpha){
#ifdef B_UNIQUE_SIGNED_CARRIER
    if(signed_bytes&&std::getenv("LOWMEM_UNIQUE_VECTOR")&&std::atoi(std::getenv("LOWMEM_UNIQUE_VECTOR")))return build<float>(raw,n,d,alpha);
#endif
    return signed_bytes?build<int8_t>(raw,n,d,alpha):build<float>(raw,n,d,alpha);
}

B64Seed build_with_evidence(const float* x,size_t n,size_t d,B64Seed& evidence,double alpha){evidence_sink=&evidence;auto g=captured_b64(x,n,d,std::getenv("LOWMEM_INPUT_DTYPE")&&std::string(std::getenv("LOWMEM_INPUT_DTYPE"))=="int8",alpha);evidence_sink=nullptr;return g;}

// SPDX-License-Identifier: MIT
// LowMem-PiPNN: measured V16 construction and FP16 serving.
#include "lowmem/support.hpp"
#include "lowmem/replay.hpp"
#include "lowmem/capture.hpp"
#include "lowmem/profile.hpp"
#include <rabitqlib/quantization/rabitq.hpp>
#include <iomanip>
#include <malloc.h>
#include <sstream>
using Clock=std::chrono::steady_clock;
using lowmem::Meta;
static float primary_factor_scale=1;
struct Matrix {uint32_t n=0,d=0;std::vector<float> x;};
static Matrix load_matrix(const char* path,size_t count,const std::string& type){
    std::ifstream f(path,std::ios::binary);Matrix m;uint32_t available;
    f.read(reinterpret_cast<char*>(&available),4);f.read(reinterpret_cast<char*>(&m.d),4);
    m.n=count?count:available;
    if(!f||!m.n||m.n>available||!m.d)throw std::runtime_error("invalid matrix prefix");
    m.x.resize(size_t(m.n)*m.d);
    if(type=="float32")f.read(reinterpret_cast<char*>(m.x.data()),m.x.size()*4);
    else if(type=="int8"||type=="uint8"){
        std::vector<uint8_t> block(1<<20);
        for(size_t begin=0;begin<m.x.size();begin+=block.size()){
            size_t size=std::min(block.size(),m.x.size()-begin);
            f.read(reinterpret_cast<char*>(block.data()),size);
            for(size_t i=0;i<size;++i)m.x[begin+i]=type=="int8"?float(int8_t(block[i])):float(block[i]);
        }
    }else throw std::runtime_error("unknown input dtype");
    if(!f)throw std::runtime_error("truncated input matrix");return m;
}
static std::vector<uint32_t> load_truth(const char* path,size_t n){
    std::ifstream f(path,std::ios::binary);uint32_t count,k;f.read(reinterpret_cast<char*>(&count),4);f.read(reinterpret_cast<char*>(&k),4);
    if(!f||!count||k<10)throw std::runtime_error("GT shape mismatch");n=std::min<size_t>(n,count);std::vector<uint32_t> all(n*k),top(n*10);
    f.read(reinterpret_cast<char*>(all.data()),all.size()*4);if(!f)throw std::runtime_error("GT truncated");
    for(size_t i=0;i<n;++i)std::copy_n(all.data()+i*k,10,top.data()+i*10);return top;
}
struct ResidentInput {uint32_t n=0,d=0;OwnedPages<float> data;};
static ResidentInput resident_load(const char* path,size_t count){
 std::ifstream f(path,std::ios::binary);uint32_t available,d;f.read(reinterpret_cast<char*>(&available),4);f.read(reinterpret_cast<char*>(&d),4);if(!f||count>available||!count)throw std::runtime_error("base header");ResidentInput x;x.n=count;x.d=d;x.data=OwnedPages<float>(count*d);const char* dtype=std::getenv("LOWMEM_INPUT_DTYPE");
 if(!dtype||std::string(dtype)=="float32")f.read(reinterpret_cast<char*>(x.data.data()),x.data.count*4);
 else if(std::string(dtype)=="uint8"||std::string(dtype)=="int8"){const bool signed_input=std::string(dtype)=="int8";std::vector<uint8_t> block(1<<20);for(size_t begin=0;begin<x.data.count;begin+=block.size()){size_t len=std::min(block.size(),x.data.count-begin);f.read(reinterpret_cast<char*>(block.data()),len);for(size_t i=0;i<len;++i)x.data[begin+i]=signed_input?float(int8_t(block[i])):float(block[i]);}}
 else throw std::runtime_error("unsupported base dtype");
 if(!f)throw std::runtime_error("truncated base");return x;
}
static std::pair<std::vector<float>,uint32_t> entry(const float* raw,size_t n,size_t d,size_t threads){
 const bool ip=handoff_metric()==rabitqlib::METRIC_IP;auto centroid=rabitqlib::compute_centroid(raw,n,d,threads);float best=std::numeric_limits<float>::max();uint32_t ep=0;
#pragma omp parallel num_threads(threads)
 {float local=std::numeric_limits<float>::max();uint32_t lid=0;
#pragma omp for schedule(static)
  for(size_t u=0;u<n;++u){float dist=ip?rabitqlib::dot_product_dis(raw+u*d,centroid.data(),d):rabitqlib::euclidean_sqr(raw+u*d,centroid.data(),d);if(dist<local||(dist==local&&u<lid)){local=dist;lid=u;}}
#pragma omp critical(b_entry)
  {if(local<best||(local==best&&lid<ep)){best=local;ep=lid;}}}
 return {std::move(centroid),ep};
}
static size_t hwm(){std::ifstream f("/proc/self/status");std::string k,line;size_t v;while(f>>k){if(k=="VmHWM:"){f>>v;return v*1024;}std::getline(f,line);}return 0;}
static double elapsed(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
static void phase(const char* s){std::cout<<"PHASE name="<<s<<" rss="<<lowmem::memory_rss()<<" hwm="<<hwm()<<"\n"<<std::flush;}
static std::vector<size_t> grid(const std::string& x){std::vector<size_t> g;std::istringstream in(x);std::string s;while(std::getline(in,s,','))g.push_back(std::stoul(s));return g;}
static void compact_encode_reused(char* out,const float* neighbors,const float* reference,size_t p,bool ip,std::vector<uint8_t>& unpacked){
 alignas(64) float a[32],b[32],error[32];
 rabitqlib::quant::rabitq_impl::one_bit::one_bit_compact_codes<float,uint8_t,false>(neighbors,reference,32,p,unpacked.data(),a,b,error,ip?rabitqlib::METRIC_IP:rabitqlib::METRIC_L2);
 rabitqlib::fastscan::pack_codes(p,unpacked.data(),32,reinterpret_cast<uint8_t*>(out));
 if(ip)std::fill_n(a,32,1.F);if(primary_factor_scale!=1){float inverse=1.F/primary_factor_scale;for(int i=0;i<32;++i){a[i]*=inverse;b[i]*=inverse;}}
 lowmem::half_store(out+p*4,a);lowmem::half_store(out+p*4+64,b);
}
#include "lowmem/factor_scale.hpp"
#include "lowmem/query.hpp"
static int build(char** a){auto begin=Clock::now();std::string fmt="v16";uint32_t shape[2];{lowmem::File f(a[2],O_RDONLY);f.read(shape,8,0);}if(shape[0]<32||shape[0]>(1u<<28))throw std::runtime_error("base needs32..2^28 vectors");auto x=resident_load(a[2],shape[0]);double load=elapsed(begin);phase("input_ready");auto primary_scale_begin=Clock::now();primary_factor_scale=primary_choose_factor_scale(x);double primary_scale_seconds=elapsed(primary_scale_begin);auto tick=Clock::now();double initial=0,topology=0;OwnedPages<uint32_t> ids;
 B64Seed e;auto g=build_with_evidence(x.data.data(),x.n,x.d,e,1.2);initial=elapsed(tick);phase("pipnn_g0_ready");malloc_trim(0);tick=Clock::now();{IdCoupling topo(x.data.data(),x.n,x.d,lowmem_threads());topo.first(e,g);topo.reverse_and_finish(e,g);ids=std::move(topo.slots);}e={};g={};malloc_trim(0);malloc_trim(0);topology=elapsed(tick);phase("topology_ready");
 tick=Clock::now();auto [center,ep]=entry(x.data.data(),x.n,x.d,lowmem_threads());double entry_sec=elapsed(tick);lowmem::Index idx;
 tick=Clock::now();double encode_sec=0,migrate_sec=0,center_sec=0;



 Meta m;m.n=x.n;m.d=x.d;m.p=((x.d+63)/64)*64;m.ep=ep;m.ip=handoff_metric()==rabitqlib::METRIC_IP;m.block_bytes=rabitqlib::QGBatchDataMap<float>::data_bytes(m.p);m.row_bytes=m.d*4+m.block_bytes+128;
 auto rot=std::unique_ptr<rabitqlib::Rotator<float>>(rabitqlib::choose_rotator<float>(m.d,rabitqlib::RotatorType::FhtKacRotator,m.p,12345));m.rotation_bytes=rot->dump_bytes();std::vector<char> state(m.rotation_bytes);rot->save(state.data());

 rot.reset();idx.init(m,state.data(),true);idx.factor_scale=primary_factor_scale;
 center_sec=elapsed(tick);tick=Clock::now();

#pragma omp parallel num_threads(lowmem_threads())
 {std::vector<float> scratch(33*m.p);std::vector<uint8_t> unpacked(m.p*4);
#pragma omp for schedule(static)
  for(size_t first=0;first<m.n;first+=64){size_t last=std::min<uint64_t>(m.n,first+64);for(size_t u=first;u<last;++u){const auto* nb=ids.data()+u*32;const float* raw=x.data.data();for(size_t k=0;k<32;++k)idx.rot->rotate(raw+size_t(nb[k])*m.d,scratch.data()+k*m.p);idx.rot->rotate(raw+u*m.d,scratch.data()+32*m.p);compact_encode_reused(const_cast<char*>(idx.block(u)),scratch.data(),scratch.data()+32*m.p,m.p,m.ip,unpacked);std::memcpy(const_cast<uint32_t*>(idx.ids(u)),nb,128);}ids.retire(first*32,last*32);}}
 ids.reset();
 encode_sec=elapsed(tick);phase("original_code_final_pools_ready");tick=Clock::now();

 if(fmt=="v16"){
  float* source=x.data.pointer;size_t old_bytes=x.data.bytes;idx.rows.adopt(reinterpret_cast<char*>(source),m.n*idx.vector_bytes);x.data.pointer=nullptr;x.data.count=x.data.bytes=0;
  size_t unit=size_t(4096)/std::gcd(size_t(4096),size_t(m.d*4));size_t chunk=unit*std::max<size_t>(1,(16<<20)/(m.d*4*unit));size_t first_rows=std::min<uint64_t>(m.n,chunk);std::vector<float> first_copy(first_rows*m.d);std::memcpy(first_copy.data(),source,first_copy.size()*4);
#pragma omp parallel num_threads(lowmem_threads())
  {for(size_t first=0;first<m.n;first+=chunk){size_t last=std::min<uint64_t>(m.n,first+chunk);
#pragma omp for schedule(static)
    for(size_t u=first;u<last;++u)idx.store_vector(u,first==0?first_copy.data()+u*m.d:source+u*m.d);
#pragma omp single
    {size_t lo=std::max(first*m.d*4,last*idx.vector_bytes),hi=last*m.d*4;lo=(lo+4095)/4096*4096;hi=hi/4096*4096;if(hi>lo&&madvise(reinterpret_cast<char*>(source)+lo,hi-lo,MADV_DONTNEED))throw std::runtime_error("in-place dead page retirement");}
   }}
  std::vector<float>().swap(first_copy);char* before=idx.rows.ptr;void* resized=std::realloc(idx.rows.ptr,idx.rows.bytes);if(!resized)throw std::bad_alloc();idx.rows.ptr=static_cast<char*>(resized);std::cout<<"INPLACE_VECTOR old_capacity_bytes="<<old_bytes<<" new_requested_bytes="<<idx.rows.bytes<<" shrink_moved="<<(before!=idx.rows.ptr)<<" original_input_last_read_before_encoding_end=1\n"<<std::flush;
 }

 std::vector<float>().swap(center);malloc_trim(0);migrate_sec=elapsed(tick);phase("index_ready");std::cout<<"BUILD arm="<<fmt<<" variant=split total_seconds="<<elapsed(begin)<<" load_seconds="<<load<<" prefix_seconds="<<initial<<" topology_seconds="<<topology<<" entry_seconds="<<entry_sec<<" center_seconds="<<center_sec<<" encode_seconds="<<encode_sec<<" migrate_seconds="<<migrate_sec<<" payload="<<idx.payload()<<" ready_rss="<<lowmem::memory_rss()<<" build_hwm="<<hwm()<<" original_fp32_code=1 double_final_index=0 layout=separate_columns ownership_transfer=1\n"<<std::flush;
 std::cout<<"SPACE compact_index_object_bytes="<<sizeof(lowmem::Index)<<" embedded_meta_bytes="<<sizeof(lowmem::Meta)<<" vector_logical_bytes="<<idx.rows.bytes<<" vector_malloc_usable_bytes="<<malloc_usable_size(idx.rows.ptr)<<" code_factor_ID_logical_bytes="<<idx.codes.bytes<<" code_factor_ID_malloc_usable_bytes="<<malloc_usable_size(idx.codes.ptr)<<" rotation_serialized_bytes="<<idx.rot->dump_bytes()<<" metadata_is_embedded=1"<<std::endl;
 double primary_total=elapsed(begin);std::cout<<"END_TO_END method=main_v16 wall_seconds="<<primary_total-load<<" load_seconds="<<load<<" total_with_input_seconds="<<primary_total<<" factor_scale="<<primary_factor_scale<<" factor_scale_seconds="<<primary_scale_seconds<<" peak_vm_hwm_bytes="<<hwm()<<" ready_rss_bytes="<<lowmem::memory_rss()<<" persistent_bytes="<<idx.payload()<<" build_threads="<<lowmem_threads()<<" index_writes=0 n="<<idx.m.n<<" d="<<idx.m.d<<"\n"<<std::flush;if(std::getenv("LOWMEM_QUERY"))main_v16_search(idx);else if(!std::getenv("LOWMEM_NO_SAVE")){lowmem::save_index(idx,a[3]);}return 0;}
int main(int argc,char** a){try{
 if(!__builtin_cpu_supports("avx512f")||!__builtin_cpu_supports("avx512bw")||!__builtin_cpu_supports("avx512dq")||!__builtin_cpu_supports("avx512vl")||!__builtin_cpu_supports("f16c"))throw std::runtime_error("LowMem-PiPNN requires AVX-512 F/BW/DQ/VL and F16C");
 std::cout<<std::setprecision(12);
 if(argc==4&&std::string(a[1])=="build")return build(a);
 if(argc==3&&std::string(a[1])=="search"){lowmem::Index index;lowmem::load_index(index,a[2]);main_v16_search(index);return 0;}
 throw std::runtime_error("use scripts/run.py --help");
}catch(const std::exception& e){std::cerr<<"LOWMEM_ERROR "<<e.what()<<"\n";return 1;}}

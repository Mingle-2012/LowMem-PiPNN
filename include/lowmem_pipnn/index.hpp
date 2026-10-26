// SPDX-License-Identifier: Apache-2.0
// FP16 serving adaptation of SymphonyQG / RaBitQ-Library. See THIRD_PARTY_NOTICES.md.
#pragma once
#include <rabitqlib/utils/rotator.hpp>
#include <rabitqlib/index/query.hpp>
#include <rabitqlib/utils/buffer.hpp>
#include <rabitqlib/utils/visited_set.hpp>
#include <rabitqlib/utils/memory.hpp>
#include <rabitqlib/utils/bitops.hpp>
#include <rabitqlib/fastscan/fastscan.hpp>
#include <immintrin.h>
#include <fcntl.h>
#include <unistd.h>
#include <sched.h>
#include <sys/mman.h>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <array>
#include <numeric>
#include <cmath>
#include <memory>
namespace lowmem {
using namespace rabitqlib;
struct Memory {
 char* ptr=nullptr;size_t bytes=0;
 Memory()=default;~Memory(){reset();}Memory(const Memory&)=delete;
 void allocate(size_t n){reset();bytes=n;size_t cap=(n+(1u<<22)-1)/(1u<<22)*(1u<<22);ptr=static_cast<char*>(memory::aligned_allocate_bytes(1u<<22,cap));if(!ptr)throw std::bad_alloc();memory::advise_huge_pages(ptr,cap);}
 void adopt(char* p,size_t n){reset();ptr=p;bytes=n;}
 void reset(){if(ptr)memory::aligned_deallocate(ptr);ptr=nullptr;bytes=0;}
};
struct File {
 int fd;File(const std::string& p,int flags):fd(open(p.c_str(),flags,0600)){if(fd<0)throw std::runtime_error("cannot open: "+p);}~File(){close(fd);}
 void read(void* dst,size_t n,size_t at)const{char* x=(char*)dst;while(n){ssize_t z=pread(fd,x,n,at);if(z<=0)throw std::runtime_error("truncated artifact");x+=z;at+=z;n-=z;}}
 void write(const void* src,size_t n)const{const char* x=(const char*)src;while(n){ssize_t z=::write(fd,x,n);if(z<=0)throw std::runtime_error("write failed");x+=z;n-=z;}}
};
struct Pin {cpu_set_t saved;int cpu=0;Pin(){if(sched_getaffinity(0,sizeof(saved),&saved))throw std::runtime_error("affinity");while(cpu<CPU_SETSIZE&&!CPU_ISSET(cpu,&saved))++cpu;cpu_set_t one;CPU_ZERO(&one);CPU_SET(cpu,&one);if(sched_setaffinity(0,sizeof(one),&one))throw std::runtime_error("pin");}~Pin(){sched_setaffinity(0,sizeof(saved),&saved);}};
struct Meta {uint64_t magic=0x4843514732303236ULL,n=0,d=0,p=0,ep=0,ip=0,block_bytes=0,row_bytes=0,rotation_bytes=0;};
inline size_t memory_rss(){std::ifstream f("/proc/self/status");std::string k,line;size_t n;while(f>>k){if(k=="VmRSS:"){f>>n;return n*1024;}std::getline(f,line);}return 0;}
inline void half_store(char* dst,const float* src){for(int i=0;i<32;i+=8)_mm_storeu_si128((__m128i*)(dst+2*i),_mm256_cvtps_ph(_mm256_loadu_ps(src+i),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC));}
inline uint32_t half_scan_mask(const char* src,size_t p,const BatchQuery<float>& q,float* out,float threshold,float factor_scale=1){
 std::array<int32_t,32> accu{};fastscan::accumulate((const uint8_t*)src,q.lut(),accu.data(),p);auto delta=_mm512_set1_ps(q.delta()),vl=_mm512_set1_ps(q.sum_vl_lut()),k1=_mm512_set1_ps(q.k1xsumq()),g=_mm512_set1_ps(q.g_add()),th=_mm512_set1_ps(threshold);uint32_t mask=0;
 for(int j=0;j<32;j+=16){auto a=_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(src+p*4+j*2)));auto b=_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(src+p*4+64+j*2)));if(factor_scale!=1){auto scale=_mm512_set1_ps(factor_scale);a=_mm512_mul_ps(a,scale);b=_mm512_mul_ps(b,scale);}auto i=_mm512_cvtepi32_ps(_mm512_loadu_si512(accu.data()+j));auto t=_mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(delta,i),vl),k1);auto score=_mm512_add_ps(_mm512_add_ps(a,g),_mm512_mul_ps(b,t));_mm512_storeu_ps(out+j,score);mask|=uint32_t(_mm512_cmp_ps_mask(score,th,_CMP_NGT_UQ))<<j;}return mask;
}
inline float half_l2_score(const float* q,const char* data,size_t d){
 auto add=[&](size_t j,__m512 acc){auto x=_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(data+j*2)));auto z=_mm512_sub_ps(_mm512_loadu_ps(q+j),x);return _mm512_fmadd_ps(z,z,acc);};
 __m512 s0=_mm512_setzero_ps(),s1=s0,s2=s0,s3=s0;size_t j=0;for(;j+64<=d;j+=64){s0=add(j,s0);s1=add(j+16,s1);s2=add(j+32,s2);s3=add(j+48,s3);}if(j+16<=d){s0=add(j,s0);j+=16;}if(j+16<=d){s1=add(j,s1);j+=16;}if(j+16<=d){s2=add(j,s2);j+=16;}float total=_mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(s0,s1),_mm512_add_ps(s2,s3)));for(;j<d;++j){uint16_t z;std::memcpy(&z,data+j*2,2);float x=_cvtsh_ss(z);total=std::fma(q[j]-x,q[j]-x,total);}return total;
}
struct Index {
 float factor_scale=1;Meta m;size_t vector_bytes=0,cb=0,bb=0,stride=0;Memory rows,codes;std::unique_ptr<Rotator<float>> rot;
 const char* row(uint32_t u)const{return rows.ptr+size_t(u)*vector_bytes;}
 const char* block(uint32_t u)const{return codes.ptr+size_t(u)*(bb+128);}
 const uint32_t* ids(uint32_t u)const{return (const uint32_t*)(block(u)+bb);}
 void init(const Meta& meta,const char* state,bool borrow=false){factor_scale=1;m=meta;if(m.ip||m.n<32||m.n>(1u<<28)||m.d==0||m.ep>=m.n||m.p!=((m.d+63)/64)*64)throw std::runtime_error("unsupported L2 R32 metadata");cb=m.p*4;bb=cb+128;vector_bytes=m.d*2;stride=vector_bytes+bb+128;if(!borrow)rows.allocate(m.n*vector_bytes);codes.allocate(m.n*(bb+128));rot.reset(choose_rotator<float>(m.d,RotatorType::FhtKacRotator,m.p,12345));rot->load(state);}
 void store_vector(uint32_t u,const float* x){char* dst=rows.ptr+size_t(u)*vector_bytes;size_t j=0;for(;j+8<=m.d;j+=8)_mm_storeu_si128((__m128i*)(dst+j*2),_mm256_cvtps_ph(_mm256_loadu_ps(x+j),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC));for(;j<m.d;++j){uint16_t z=_cvtss_sh(x[j],_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);std::memcpy(dst+j*2,&z,2);}}
 float vertex_score(const float* q,uint32_t u)const{return half_l2_score(q,row(u),m.d);}
 void prefetch(uint32_t u)const{if(u>=m.n)return;memory::mem_prefetch_l2(row(u),std::min<size_t>(10,(vector_bytes+63)/64));memory::mem_prefetch_l2(block(u),(bb+128+63)/64);}
 size_t payload()const{return rows.bytes+codes.bytes+rot->dump_bytes()+sizeof(Meta)+sizeof(factor_scale);}
};
struct Scratch {size_t ef;std::vector<float> rotated,est,lut;BatchQuery<float> batch;buffer::SearchBuffer<float> beam,top;VisitedSet visited;Scratch(size_t n,size_t p,size_t l):ef(l),rotated(p),est(32),lut(p*4),beam(l),top(10){if(l<10)throw std::runtime_error("search L must be >=10");visited.initialize(n,n/10);}};
inline void search(const Index& v,const float* query,uint32_t* result,float* distances,Scratch& s){
 v.rot->rotate(query,s.rotated.data());s.batch.reset(s.rotated.data(),v.m.p,s.lut.data(),METRIC_L2);s.beam.clear();s.top.clear();s.visited.clear();s.beam.insert(v.m.ep,std::numeric_limits<float>::max());
 while(s.beam.has_next()){uint32_t u=s.beam.pop();if(s.visited.get(u))continue;s.visited.set(u);float vertex=v.vertex_score(query,u);s.batch.set_g_add(vertex);const auto* ids=v.ids(u);uint32_t mask=half_scan_mask(v.block(u),v.m.p,s.batch,s.est.data(),s.beam.top_dist(),v.factor_scale);
  while(mask){size_t k=bitops::countr_zero32(mask);mask&=mask-1;uint32_t id=ids[k];float ds=s.est[k];if(s.beam.is_full(ds)||s.visited.get(id))continue;s.beam.insert(id,ds);if(s.beam.has_next())v.prefetch(s.beam.next_id());}s.top.insert(u,vertex);}
 if(!s.top.is_full()){auto& records=s.top.data();std::vector<AnnCandidate<float>> copy(records.begin(),records.begin()+s.top.size());for(auto a:copy){auto* ids=v.ids(a.id);for(size_t k=0;k<32;++k)if(!s.visited.get(ids[k])){s.visited.set(ids[k]);s.top.insert(ids[k],v.vertex_score(query,ids[k]));}if(s.top.is_full())break;}}
 if(s.top.size()!=10)throw std::runtime_error("underfilled top10");s.top.copy_results(result,distances);
}
inline void save_index(const Index& idx,const std::string& prefix){File meta(prefix+".meta",O_CREAT|O_TRUNC|O_WRONLY),v(prefix+".vec",O_CREAT|O_TRUNC|O_WRONLY),c(prefix+".blocks",O_CREAT|O_TRUNC|O_WRONLY);uint64_t magic=idx.factor_scale==1?0x5643303030303033ULL:0x5643303030303034ULL;uint32_t format=2,vk=1;std::vector<char> state(idx.rot->dump_bytes());idx.rot->save(state.data());meta.write(&magic,8);meta.write(&idx.m,sizeof(idx.m));meta.write(&format,4);meta.write(&vk,4);meta.write(state.data(),state.size());if(idx.factor_scale!=1)meta.write(&idx.factor_scale,4);v.write(idx.rows.ptr,idx.rows.bytes);c.write(idx.codes.ptr,idx.codes.bytes);fsync(v.fd);fsync(c.fd);fsync(meta.fd);}
inline void load_index(Index& idx,const std::string& prefix){File meta(prefix+".meta",O_RDONLY);uint64_t magic;Meta m;uint32_t format,vk;meta.read(&magic,8,0);meta.read(&m,sizeof(m),8);meta.read(&format,4,8+sizeof(m));meta.read(&vk,4,12+sizeof(m));if((magic!=0x5643303030303033ULL&&magic!=0x5643303030303034ULL)||format!=2||vk!=1)throw std::runtime_error("expected LowMem FP16 split index");if(m.rotation_bytes>1<<24)throw std::runtime_error("rotation metadata");std::vector<char> state(m.rotation_bytes);meta.read(state.data(),state.size(),16+sizeof(m));idx.init(m,state.data());if(magic==0x5643303030303034ULL){meta.read(&idx.factor_scale,4,16+sizeof(m)+m.rotation_bytes);int e;float f=std::frexp(idx.factor_scale,&e);if(!std::isfinite(idx.factor_scale)||idx.factor_scale<1||f!=.5F)throw std::runtime_error("invalid factor scale");}for(auto item:{std::pair<const char*,Memory*>(".vec",&idx.rows),{".blocks",&idx.codes}}){if(std::filesystem::file_size(prefix+item.first)!=item.second->bytes)throw std::runtime_error("index file length");File f(prefix+item.first,O_RDONLY);f.read(item.second->ptr,item.second->bytes,0);}}
}

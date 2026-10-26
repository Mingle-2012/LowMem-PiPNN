#pragma once
#include "evidence_view.hpp"
#include <atomic>
#include <iomanip>
#include <unordered_set>
#include <set>
namespace bpre {
inline constexpr size_t candidate_cap=64;
inline int mode=std::getenv("LOWMEM_PRE_MODE")?std::stoi(std::getenv("LOWMEM_PRE_MODE")):0;
inline bool check=std::getenv("LOWMEM_PRE_CHECK")!=nullptr;
inline int optimization=std::getenv("LOWMEM_PRE_OPT")?std::stoi(std::getenv("LOWMEM_PRE_OPT")):0;
// Same epoch/open-addressing scheme as brev::Workspace, without its downstream
// List/metric dependencies. unique retains insertion order across table growth.
struct EpochSeen {
 std::vector<uint64_t> table;std::vector<uint32_t> unique;uint32_t epoch=0;
 void reset(){unique.clear();if(++epoch==0){std::fill(table.begin(),table.end(),0);epoch=1;}if(table.empty())table.resize(2048,0);}
 size_t slot(uint32_t id)const{uint64_t h=uint64_t(id)*0x9e3779b97f4a7c15ULL;h^=h>>32;return h&(table.size()-1);}
 void grow(){auto old=std::move(table);table.assign(old.size()*2,0);
 for(auto id:unique){size_t k=slot(id);while(uint32_t(table[k]>>32)==epoch)k=(k+1)&(table.size()-1);table[k]=(uint64_t(epoch)<<32)|id;
 }}
 bool insert(uint32_t id){if((unique.size()+1)*2>=table.size())grow();size_t k=slot(id);while(uint32_t(table[k]>>32)==epoch){if(uint32_t(table[k])==id){
 return false;}k=(k+1)&(table.size()-1);}table[k]=(uint64_t(epoch)<<32)|id;unique.push_back(id);
 return true;}
};
inline const B64Seed* source=nullptr;
inline B64Seed reverse;
struct alignas(64) Counts {uint64_t scored=0,kept=0,enumerated=0;};
inline std::vector<Counts> counts;
inline std::chrono::steady_clock::time_point start;
inline void prepare(const B64Seed& e){
 if(check&&optimization){EpochSeen test;test.reset();for(uint32_t i=0;i<5000;++i){if(!test.insert(i)||test.insert(i))throw std::runtime_error("epoch grow/duplicate failure");}for(uint32_t i=0;i<5000;++i)if(test.unique[i]!=i)throw std::runtime_error("epoch insertion order failure");test.reset();if(!test.insert(0)||test.insert(0))throw std::runtime_error("epoch reset failure");test.epoch=UINT32_MAX;test.reset();if(test.epoch!=1||!test.insert(0)||test.insert(0))throw std::runtime_error("epoch wrap failure");std::cout<<"PRE_EPOCH_VALIDATION grow_duplicate_order_reset_wrap=passed"<<std::endl;}
 source=&e;reverse={};counts.assign(parlay::num_workers(),{});start=std::chrono::steady_clock::now();
 bool need_reverse=mode&2;
 if(!need_reverse)return;size_t n=e.offsets.size()-1;reverse.offsets.assign(n+1,0);
 parlay::parallel_for(0,n,[&](size_t u){for(size_t k=e.offsets[u];k<e.offsets[u+1];++k)__atomic_fetch_add(&reverse.offsets[e.neighbors[k]+1],size_t(1),__ATOMIC_RELAXED);});
 for(size_t u=0;u<n;++u)reverse.offsets[u+1]+=reverse.offsets[u];reverse.neighbors.resize(reverse.offsets.back());auto cursor=reverse.offsets;
 parlay::parallel_for(0,n,[&](size_t u){for(size_t k=e.offsets[u];k<e.offsets[u+1];++k){auto v=e.neighbors[k];auto j=__atomic_fetch_add(&cursor[v],size_t(1),__ATOMIC_RELAXED);reverse.neighbors[j]=u;}});
}
struct Item {uint32_t point_id;float norm;};
template<class Points> float discovery_distance(size_t u,uint32_t v,Points& points){
 return brq::estimate(v);
}
inline thread_local const std::vector<Item>* block_result=nullptr;
template<class Graph,class Points> const std::vector<Item>& pool(size_t u,const Graph& graph,Points& points){
 if(block_result)return *block_result;
 brq::query(u);
 thread_local std::vector<Item> result,extra;thread_local std::unordered_set<uint32_t> seen;
 thread_local EpochSeen flat;
 if(optimization<0||optimization>3)throw std::runtime_error("pre optimization must be0/1/2/3");
 result.clear();extra.clear();if(optimization)flat.reset();else seen.clear();
 auto insert=[&](uint32_t v){return optimization?flat.insert(v):seen.insert(v).second;};
 insert(u);
 const auto& e=*source;
 uint64_t enumerated=0;
 for(auto c:graph.Neighbors(u)){result.push_back({c.point_id,c.norm});insert(c.point_id);}
 size_t base_end=flat.unique.size();
 auto add=[&](uint32_t v){++enumerated;if(insert(v)&&optimization!=2)extra.push_back({v,discovery_distance(u,v,points)});};
 if(mode&1)for(size_t k=e.offsets[u];k<e.offsets[u+1];++k){auto v=e.neighbors[k];for(size_t j=e.offsets[v];j<e.offsets[v+1];++j){auto id=e.neighbors[j];
 add(id);
 }}
 if(mode&2)for(size_t k=reverse.offsets[u];k<reverse.offsets[u+1];++k)add(reverse.neighbors[k]);
 if(optimization==2){extra.reserve(flat.unique.size()-base_end);for(size_t k=base_end;k<flat.unique.size();++k){auto v=flat.unique[k];extra.push_back({v,points[u].distance(points[v])});}}
 auto& c=counts[parlay::worker_id()];c.scored+=extra.size();c.enumerated+=enumerated;
 auto rank=[](auto a,auto b){return a.norm<b.norm||(a.norm==b.norm&&a.point_id<b.point_id);};
 if(extra.size()>candidate_cap){std::nth_element(extra.begin(),extra.begin()+candidate_cap,extra.end(),rank);extra.resize(candidate_cap);}c.kept+=extra.size();
 if(check&&u%1024==0){std::set<uint32_t> base;for(auto c:graph.Neighbors(u))base.insert(c.point_id);base.insert(u);auto oracle=base;
  if(mode&1)for(size_t k=e.offsets[u];k<e.offsets[u+1];++k){auto v=e.neighbors[k];oracle.insert(e.neighbors.begin()+e.offsets[v],e.neighbors.begin()+e.offsets[v+1]);}
  if(mode&2)oracle.insert(reverse.neighbors.begin()+reverse.offsets[u],reverse.neighbors.begin()+reverse.offsets[u+1]);
  std::vector<Item> expected;for(auto v:oracle)if(!base.count(v))expected.push_back({v,discovery_distance(u,v,points)});std::sort(expected.begin(),expected.end(),rank);if(expected.size()>candidate_cap)expected.resize(candidate_cap);auto actual=extra;std::sort(actual.begin(),actual.end(),rank);if(expected.size()!=actual.size())throw std::runtime_error("pre pool oracle size");for(size_t k=0;k<expected.size();++k)if(expected[k].point_id!=actual[k].point_id||std::memcmp(&expected[k].norm,&actual[k].norm,4))throw std::runtime_error("pre pool oracle ID/FP32 mismatch");
 }
 std::sort(extra.begin(),extra.end(),[](auto a,auto b){return a.point_id<b.point_id;});
 for(auto& x:extra)x.norm=points[u].distance(points[x.point_id]);
 result.insert(result.end(),extra.begin(),extra.end());
 // On every node in full-data validation, replay the OLD unordered_set +
 // interleaved scoring path on this exact E/source row. Compare submitted order
 // and FP32 bits, not only sets; no cross-build equality assumption is needed.
 if(check&&optimization){
  thread_local std::unordered_set<uint32_t> reference_seen;thread_local std::vector<Item> reference,reference_extra;
  reference_seen.clear();reference.clear();reference_extra.clear();reference_seen.insert(u);
  for(auto x:graph.Neighbors(u)){reference.push_back({x.point_id,x.norm});reference_seen.insert(x.point_id);}
  auto add_reference=[&](uint32_t v){if(reference_seen.insert(v).second)reference_extra.push_back({v,discovery_distance(u,v,points)});};
  if(mode&1)for(size_t k=e.offsets[u];k<e.offsets[u+1];++k){auto v=e.neighbors[k];for(size_t j=e.offsets[v];j<e.offsets[v+1];++j)add_reference(e.neighbors[j]);}
  if(mode&2)for(size_t k=reverse.offsets[u];k<reverse.offsets[u+1];++k)add_reference(reverse.neighbors[k]);
  if(reference_extra.size()>candidate_cap){std::nth_element(reference_extra.begin(),reference_extra.begin()+candidate_cap,reference_extra.end(),rank);reference_extra.resize(candidate_cap);}
  std::sort(reference_extra.begin(),reference_extra.end(),[](auto a,auto b){return a.point_id<b.point_id;});
  for(auto& x:reference_extra)x.norm=points[u].distance(points[x.point_id]);
  reference.insert(reference.end(),reference_extra.begin(),reference_extra.end());
  if(reference.size()!=result.size())throw std::runtime_error("pre exact old-path size mismatch");
  for(size_t k=0;k<result.size();++k)if(reference[k].point_id!=result[k].point_id||std::memcmp(&reference[k].norm,&result[k].norm,4))throw std::runtime_error("pre exact old-path ordered ID/FP32 mismatch");
 }
 return result;
}
inline void finish(){Counts total;for(auto c:counts){total.scored+=c.scored;total.kept+=c.kept;total.enumerated+=c.enumerated;}
 std::cout.unsetf(std::ios::floatfield);std::cout<<std::setprecision(12)<<"PRE_STAGE mode="<<mode<<" seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<" unique_new_scores="<<total.scored<<" selected_new="<<total.kept<<" enumerated="<<total.enumerated<<" reverse_payload_bytes="<<reverse.neighbors.capacity()*4+reverse.offsets.capacity()*8<<" cap_per_node="<<candidate_cap<<" immutable_E=1"<<std::endl;
 std::cout<<"PRE_EXACT_OPT mode="<<optimization<<" all_node_old_path_check="<<(check&&optimization?1:0)<<std::endl;
 std::cout<<"PRE_RABITQ_SCORE approximate_evaluations="<<total.scored<<" native_selected_evaluations="<<total.kept<<" selection=approx_top"<<candidate_cap<<" native_refine_after_selection=1"<<std::endl;
 brq::release();
 reverse={};source=nullptr;
}
}

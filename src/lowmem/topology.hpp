#pragma once
#include "working_point.hpp"
#include "support.hpp"
#include "capture.hpp"
#include "../../third_party/rabitq/include/rabitqlib/utils/space.hpp"
using Candidate=rabitqlib::AnnCandidate<float>;
using List=rabitqlib::symqg::CandidateList;
inline rabitqlib::MetricType handoff_metric(){const char* m=std::getenv("LOWMEM_METRIC");return m&&std::string(m)=="IP"?rabitqlib::METRIC_IP:rabitqlib::METRIC_L2;}
struct RawTopology {
 const float* x;size_t n,d;size_t threads;
 bool ip=handoff_metric()==rabitqlib::METRIC_IP;std::vector<List> kept,rejected;
 RawTopology(const float* a,size_t N,size_t D,size_t T,bool allocate=true):x(a),n(N),d(D),threads(T),kept(allocate?N:0),rejected(allocate?N:0){
 }
 float distance(uint32_t u,uint32_t v)const{
  return ip?rabitqlib::dot_product_dis(x+u*d,x+v*d,d):(rabitqlib::signed_int8_input()?rabitqlib::int8_l2_carrier(x+u*d,x+v*d,d):rabitqlib::euclidean_sqr(x+u*d,x+v*d,d));
 }
 void prune(const List& pool,List& result,List& evidence,uint32_t source=UINT32_MAX)const{
  if(pool.empty())return;result.clear();if(pool.size()<=lowmem_degree()){result=pool;return;}
  std::vector<bool> removed(pool.size(),false);
  for(size_t j=0;j<pool.size()&&result.size()<lowmem_degree();++j){if(removed[j])continue;result.push_back(pool[j]);
   for(size_t k=j+1;k<pool.size();++k){if(removed[k])continue;if(distance(pool[j].id,pool[k].id)<pool[k].distance){
    if(evidence.size()<300)evidence.push_back(pool[k]);removed[k]=true;}}
  }
 }
 void add(const List& row,const List& evidence,List& result,float threshold)const{
  result=row;std::unordered_set<uint32_t> ids;ids.reserve(lowmem_degree());for(auto e:row)ids.insert(e.id);
  size_t k=0;while(result.size()<lowmem_degree()&&k<evidence.size()){
   auto cur=evidence[k];if(ids.count(cur.id))break;bool occlude=false;
   for(auto e:result){if(e.distance>cur.distance)break;float djk=distance(cur.id,e.id);
    float cosine=(cur.distance+e.distance-djk)/(2*std::sqrt(e.distance*cur.distance));if(cosine>threshold){occlude=true;break;}}
   if(!occlude){result.push_back(cur);ids.insert(cur.id);std::sort(result.begin(),result.end());}++k;
  }
 }
 void complete(size_t u,List& row,List& evidence)const{
  if(row.size()>=lowmem_degree())return;
  std::sort(evidence.begin(),evidence.end());List result,successful;float left=.5F,right=1.F;size_t iter=0;
  while(iter++<rabitqlib::symqg::kMaxBsIter){float mid=(left+right)/2;add(row,evidence,result,mid);if(result.size()<lowmem_degree())left=mid;else{right=mid;successful=result;}}
  if(!successful.empty())result.swap(successful);else add(row,evidence,result,right);
  if(result.size()<lowmem_degree()){
   std::mt19937 rng(12345^uint32_t(u)^0x80000000U);std::uniform_int_distribution<uint32_t> distribution(0,uint32_t(n-1));std::unordered_set<uint32_t> ids;for(auto e:result)ids.insert(e.id);
   while(result.size()<lowmem_degree()){auto v=distribution(rng);if(v!=u&&!ids.count(v)){result.emplace_back(v,distance(u,v));ids.insert(v);}}}
  row=result;
 }
 void first(const B64Seed& evidence,const B64Seed& graph){
#pragma omp parallel for num_threads(threads) schedule(dynamic)
  for(size_t u=0;u<n;++u){std::vector<uint32_t> ids(evidence.neighbors.begin()+evidence.offsets[u],evidence.neighbors.begin()+evidence.offsets[u+1]);ids.insert(ids.end(),graph.neighbors.begin()+graph.offsets[u],graph.neighbors.begin()+graph.offsets[u+1]);std::sort(ids.begin(),ids.end());ids.erase(std::unique(ids.begin(),ids.end()),ids.end());List pool;for(auto v:ids)if(v!=u)pool.emplace_back(v,distance(u,v));std::sort(pool.begin(),pool.end());prune(pool,kept[u],rejected[u],u);}
 }
 void couple(){
#pragma omp parallel for num_threads(threads) schedule(dynamic)
  for(size_t u=0;u<n;++u)complete(u,kept[u],rejected[u]);
 }
};

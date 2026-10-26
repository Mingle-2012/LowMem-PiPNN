#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <vector>
#include "../../../parlay/include/parlay/parallel.h"

namespace parlayANN {
// Eleven little-endian u32 fields per expansion; query_id is supplied on dump.
struct EdgeEvent {
  uint32_t node=0,degree=0,scanned=0,seen=0,same_as=0,novel=0;
  uint32_t insertions=0,distance_rejected=0,beam_rejected=0,frontier_redundant=0;
};
static_assert(sizeof(EdgeEvent)==40);
struct EdgeQuery {
  std::vector<EdgeEvent> events;
  std::vector<std::pair<uint32_t,uint32_t>> pending;
  template<class A,class B> void merge(const A& old_frontier,const B& new_frontier,size_t new_size){
    std::unordered_set<uint32_t> old,now,credited;
    for(auto x:old_frontier)old.insert(x.first);
    for(size_t j=0;j<new_size;++j)now.insert(new_frontier[j].first);
    for(auto [id,origin]:pending){
      auto& e=events[origin];
      if(!now.count(id))++e.beam_rejected;
      else if(old.count(id)||!credited.insert(id).second)++e.frontier_redundant;
      else ++e.insertions;
    }
    pending.clear();
  }
};
struct EdgeProductivity {
  bool active=false;
  std::vector<EdgeQuery> queries;
  void initialize(size_t n){
    active=std::getenv("PIPNN_EDGE_EVENTS_OUT")!=nullptr;
    if(active){queries.clear();queries.resize(n);}
  }
  template<class Stats> void dump(size_t beam,const Stats& stats){
    if(!active)return;
    std::ofstream out(std::string(std::getenv("PIPNN_EDGE_EVENTS_OUT"))+"_L"+std::to_string(beam)+".u32",std::ios::binary);
    if(!out)throw std::runtime_error("cannot open edge events");
    uint64_t total=0,pending=0;
    for(uint32_t q=0;q<queries.size();++q){
      auto& trace=queries[q];
      if(trace.events.size()!=stats.visited[q])throw std::runtime_error("edge event count mismatch");
      uint64_t novel=0;
      for(auto e:trace.events){
        if(e.scanned!=e.seen+e.same_as+e.novel)throw std::runtime_error("edge partition mismatch");
        novel+=e.novel;
        if(!out.write(reinterpret_cast<char*>(&q),4)||!out.write(reinterpret_cast<char*>(&e),sizeof(e)))
          throw std::runtime_error("edge event write failed");
      }
      // These unquantized runs have one seed-distance plus novel evaluations.
      if(novel+1!=stats.distances[q])throw std::runtime_error("edge distance count mismatch");
      total+=trace.events.size();pending+=trace.pending.size();
    }
    std::cout<<"EDGE_PRODUCTIVITY events="<<total<<" pending_at_termination="<<pending<<'\n';
  }
};
inline EdgeProductivity edge_productivity;

// Fixed N x 10 float32 summaries: count, d1/d8/d16/d32/d48/d64, LID16/32/48.
template<class GraphH> void export_candidate_geometry(GraphH& graph,size_t n,const char* path){
  const auto start=std::chrono::steady_clock::now();
  std::ofstream out(path,std::ios::binary);
  if(!out)throw std::runtime_error("cannot open candidate geometry");
  constexpr size_t chunk=65536;
  std::vector<std::array<float,10>> rows(std::min(n,chunk));
  for(size_t begin=0;begin<n;begin+=chunk){
    size_t count=std::min(chunk,n-begin);
    parlay::parallel_for(0,count,[&](size_t j){
      std::vector<double> r;
      if(graph.capacity(begin+j)!=64)throw std::runtime_error("geometry requires Uniform B64");
      for(auto x:graph.Neighbors(begin+j))r.push_back(std::sqrt(std::max(0.0,double(float(x.norm)))));
      std::sort(r.begin(),r.end());
      auto& row=rows[j];row.fill(std::numeric_limits<float>::quiet_NaN());row[0]=r.size();
      size_t col=1;for(size_t rank:{1,8,16,32,48,64}){if(r.size()>=rank)row[col]=r[rank-1];++col;}
      for(size_t k:{16,32,48}){
        if(r.size()>=k && r[0]>0 && std::isfinite(r[k-1])){
          double sum=0;for(size_t i=0;i<k-1;++i)sum+=std::log(r[i]/r[k-1]);
          if(sum<0)row[col]=-double(k-1)/sum;
        }++col;
      }
    });
    if(!out.write(reinterpret_cast<char*>(rows.data()),count*sizeof(rows[0])))throw std::runtime_error("geometry write failed");
  }
  std::cout<<"CANDIDATE_GEOMETRY rows="<<n<<" bytes="<<n*40<<" seconds="
           <<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<'\n';
}
} // namespace parlayANN

#pragma once
#include "owned_pages.hpp"
#include "graph_rows.hpp"
#include <iomanip>

namespace brev {
inline size_t depth=std::getenv("LOWMEM_G0_REVERSE_DEPTH")?std::stoul(std::getenv("LOWMEM_G0_REVERSE_DEPTH")):1;
inline int post_mode=std::getenv("LOWMEM_POST_MODE")?std::stoi(std::getenv("LOWMEM_POST_MODE")):3;
struct Index {
 std::vector<size_t> offsets;OwnedPages<uint32_t> ids;
 template<class Graph> void build(const Graph& g,size_t threads){
  auto start=std::chrono::steady_clock::now();size_t n=graph_nodes(g);offsets.assign(n+1,0);
#pragma omp parallel for num_threads(threads) schedule(static)
  for(size_t u=0;u<n;++u){std::array<uint32_t,64> buf{};for(auto v:graph_row(g,u,buf))__atomic_fetch_add(&offsets[v+1],size_t(1),__ATOMIC_RELAXED);}
  for(size_t u=0;u<n;++u)offsets[u+1]+=offsets[u];ids=OwnedPages<uint32_t>(offsets.back());
  std::vector<size_t> cursor(offsets.begin(),offsets.end()-1);
#pragma omp parallel for num_threads(threads) schedule(static)
  for(size_t u=0;u<n;++u){std::array<uint32_t,64> buf{};for(auto v:graph_row(g,u,buf)){auto slot=__atomic_fetch_add(&cursor[v],size_t(1),__ATOMIC_RELAXED);ids[slot]=u;}}
  // No row sort: discovery is a set, extras use (distance,ID), final candidates
  // are ID-canonicalized before the original distance-only pruning sort.
  size_t max_in=0;for(size_t u=0;u<n;++u){if(cursor[u]!=offsets[u+1])throw std::runtime_error("G0 transpose scatter mismatch");max_in=std::max(max_in,offsets[u+1]-offsets[u]);}
  std::cout.unsetf(std::ios::floatfield);std::cout<<std::setprecision(12)<<"G0_REVERSE_INDEX seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<" edges="<<ids.count<<" payload_bytes="<<ids.bytes+offsets.capacity()*sizeof(size_t)<<" transient_cursor_bytes="<<cursor.capacity()*sizeof(size_t)<<" max_indegree="<<max_in<<" build_threads="<<threads<<" sorted_rows=0 builds=1"<<std::endl;
 }
};
struct Workspace {
 std::vector<uint64_t> table;std::vector<uint32_t> unique;List extra;uint32_t epoch=0;
 void reset(){unique.clear();extra.clear();if(++epoch==0){std::fill(table.begin(),table.end(),0);epoch=1;}if(table.empty())table.resize(2048,0);}
 size_t slot(uint32_t id)const{uint64_t h=uint64_t(id)*0x9e3779b97f4a7c15ULL;h^=h>>32;return h&(table.size()-1);}
 void grow(){auto old=std::move(table);table.assign(old.size()*2,0);
  for(size_t j=0;j<old.size();++j)if(uint32_t(old[j]>>32)==epoch){auto id=uint32_t(old[j]);size_t k=slot(id);while(uint32_t(table[k]>>32)==epoch)k=(k+1)&(table.size()-1);table[k]=old[j];
  }}
 bool insert(uint32_t id,uint64_t position=UINT64_MAX){if((unique.size()+1)*2>=table.size())grow();size_t k=slot(id);while(uint32_t(table[k]>>32)==epoch){if(uint32_t(table[k])==id)return false;k=(k+1)&(table.size()-1);}table[k]=(uint64_t(epoch)<<32)|id;unique.push_back(id);
  return true;}
};
inline thread_local Workspace workspace;
// Independent OpenAI variant: cache only the selected NEW candidates, not
// first-prune survivors or the complete discovery stream. No distances stored.
struct SelectedExtras {
 OwnedPages<uint32_t> ids;std::vector<uint8_t> counts;
 explicit SelectedExtras(size_t n):ids(n*64),counts(n){}
 template<class E,class G> void prepare(const E&,const G&,const Index&,size_t){}
 void save(size_t u,const List& extra){
  if(extra.size()>64)throw std::runtime_error("cached extra cap exceeded");
  counts[u]=static_cast<uint8_t>(extra.size());
  for(size_t k=0;k<extra.size();++k)ids[u*64+k]=extra[k].id;
 }
 template<class E,class G> void pool(const RawTopology& metric,size_t u,const E& e,const G& g,List& out)const{
  auto& w=workspace;w.reset();w.insert(u);
  std::array<uint32_t,64> eb{},gb{};
  for(auto v:graph_row(e,u,eb))w.insert(v);
  for(auto v:graph_row(g,u,gb))w.insert(v);
  for(size_t k=0;k<counts[u];++k)if(!w.insert(ids[u*64+k]))throw std::runtime_error("cached extra duplicate/base ID");
  out.clear();out.reserve(w.unique.size()-1);
  for(size_t k=1;k<w.unique.size();++k){auto v=w.unique[k];out.emplace_back(v,metric.distance(u,v));}
  std::sort(out.begin(),out.end(),[](auto a,auto b){return a.id<b.id;});std::sort(out.begin(),out.end());
  if(out.size()>750)out.resize(750);
 }
 void release(){ids.reset();std::vector<uint8_t>().swap(counts);}
};
template<class E,class G> inline void pool(const RawTopology& metric,size_t u,const E& e,const G& g,const Index& reverse,List& out){
 if(depth!=1&&depth!=2)throw std::runtime_error("reverse depth must be1/2");
 if(64!=64&&64!=128)throw std::runtime_error("shared extra cap must be64/128");
 auto& w=workspace;size_t base_end=0,raw=0;
 {
  w.reset();w.insert(u);
  uint64_t discovery_position=0;
  auto add=[&](uint32_t id){++raw;w.insert(id);};
  auto discover=[&](uint32_t id){++raw;w.insert(id,discovery_position++);};
  std::array<uint32_t,64> eb{},gb{},vb{};auto gr=graph_row(g,u,gb);
  for(auto v:graph_row(e,u,eb))add(v);
  for(auto v:gr)add(v);base_end=w.unique.size();
   if(post_mode&1)for(auto v:gr)for(auto id:graph_row(g,v,vb))discover(id);
   if((post_mode&2)
   )for(size_t k=reverse.offsets[u];k<reverse.offsets[u+1];++k){auto v=reverse.ids[k];discover(v);if(depth==2)for(size_t j=reverse.offsets[v];j<reverse.offsets[v+1];++j)discover(reverse.ids[j]);}
   // u -> v <- w: follow an OUT edge then the anchor's IN row. Reuse
   // the same frozen transpose; never scatter all anchor pairs into other rows.
   for(auto v:gr)for(size_t j=reverse.offsets[v];j<reverse.offsets[v+1];++j)discover(reverse.ids[j]);
 }
 {
  out.clear();out.reserve(base_end-1+64);w.extra.reserve(w.unique.size()-base_end);
  auto score=[&](uint32_t v){return metric.distance(u,v);};
  for(size_t k=1;k<base_end;++k){auto v=w.unique[k];out.emplace_back(v,score(v));}
  for(size_t k=base_end;k<w.unique.size();++k){auto v=w.unique[k];w.extra.emplace_back(v,score(v));}
  auto rank=[](auto a,auto b){return a.distance<b.distance||(a.distance==b.distance&&a.id<b.id);};
  if(w.extra.size()>64){std::nth_element(w.extra.begin(),w.extra.begin()+64,w.extra.end(),rank);w.extra.resize(64);}
  out.insert(out.end(),w.extra.begin(),w.extra.end());std::sort(out.begin(),out.end(),[](auto a,auto b){return a.id<b.id;});std::sort(out.begin(),out.end());
  if(out.size()>750)out.resize(750);
 }
}
}

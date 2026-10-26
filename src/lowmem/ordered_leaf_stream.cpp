#include "ordered_leaf_stream.hpp"
#include <Eigen/Core>
#include "../../third_party/parlay/include/parlay/parallel.h"
#include <algorithm>
#include <atomic>
#include <cfenv>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <chrono>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <unordered_set>
namespace parlayANN {float euclidian_distance(const float*,const float*,unsigned);}
namespace orderleaf {
struct Group {uint64_t begin,end;uint32_t center,left=UINT32_MAX,right=UINT32_MAX;double radius=0,norm=0;};
struct Tree {std::vector<Group> nodes;};
struct Native {const float* data;size_t d,pitch;const float* x(uint32_t u)const{return data+size_t(u)*pitch;}float score(uint32_t u,uint32_t v)const{return parlayANN::euclidian_distance(x(u),x(v),d);}double norm(uint32_t u)const{double a=0;for(size_t k=0;k<d;++k)a+=double(x(u)[k])*x(u)[k];return a;}};
static uint32_t build_tree(Tree& tree,std::vector<uint32_t>& ids,uint64_t begin,uint64_t end,const Native& v){
 uint32_t index=tree.nodes.size();tree.nodes.emplace_back();Group g;g.begin=begin;g.end=end;
 std::vector<double> mean(v.d,0);for(size_t i=begin;i<end;++i){for(size_t k=0;k<v.d;++k)mean[k]+=v.x(ids[i])[k];g.norm=std::max(g.norm,v.norm(ids[i]));}for(auto& x:mean)x/=double(end-begin);
 double closest=INFINITY;for(size_t i=begin;i<end;++i){double r=0;for(size_t k=0;k<v.d;++k){double t=double(v.x(ids[i])[k])-mean[k];r+=t*t;}if(r<closest||(r==closest&&ids[i]<g.center)){closest=r;g.center=ids[i];}}
 for(size_t i=begin;i<end;++i){double r=0;for(size_t k=0;k<v.d;++k){double t=double(v.x(ids[i])[k])-v.x(g.center)[k];r+=t*t;}g.radius=std::max(g.radius,std::sqrt(r));}
 double gd=(4*v.d+32)*0x1p-53/(1-(4*v.d+32)*0x1p-53);g.norm=std::nextafter(g.norm/(1-gd),INFINITY);g.radius=std::nextafter(g.radius/(1-gd)+32*gd*(std::sqrt(v.norm(g.center))+std::sqrt(g.norm))+double(v.d+1)*DBL_MIN,INFINITY);
 if(end-begin>32){size_t axis=0;double spread=-1;for(size_t k=0;k<v.d;++k){float lo=INFINITY,hi=-INFINITY;for(size_t i=begin;i<end;++i){lo=std::min(lo,v.x(ids[i])[k]);hi=std::max(hi,v.x(ids[i])[k]);}if(double(hi)-lo>spread){spread=double(hi)-lo;axis=k;}}
  auto mid=begin+(end-begin)/2;std::nth_element(ids.begin()+begin,ids.begin()+mid,ids.begin()+end,[&](auto a,auto b){float x=v.x(a)[axis],y=v.x(b)[axis];return x<y||(x==y&&a<b);});g.left=build_tree(tree,ids,begin,mid,v);g.right=build_tree(tree,ids,mid,end,v);
 }tree.nodes[index]=g;return index;
}
static double lower(uint32_t u,const Group& g,const Native& v){
 double r=0,nu=v.norm(u);for(size_t k=0;k<v.d;++k){double t=double(v.x(u)[k])-v.x(g.center)[k];r+=t*t;}
 double gd=(4*v.d+32)*0x1p-53/(1-(4*v.d+32)*0x1p-53);nu=std::nextafter(nu/(1-gd),INFINITY);
 if(!std::isfinite(nu)||!std::isfinite(g.norm)||nu+g.norm>0x1p60)return -INFINITY;
 double dc=std::max(0.,std::sqrt(r)/(1+gd)-32*gd*(std::sqrt(nu)+g.radius+1));double gap=std::max(0.,dc-g.radius);double e=(4*v.d+32)*0x1p-24;double delta=std::nextafter(16*e/(1-e)*(nu+g.norm)+128*double(v.d+1)*FLT_MIN,INFINITY);
 return std::nextafter(gap*gap/(1+gd)-delta,-INFINITY);
}
struct Item {uint32_t id;float distance;};
static bool rank(Item a,Item b){return a.distance<b.distance||(a.distance==b.distance&&a.id<b.id);}
struct ItemCmp {bool operator()(Item a,Item b)const{return rank(b,a);}};
struct Task {double lb;uint32_t leaf,group;};
struct TaskCmp {bool operator()(Task a,Task b)const{return a.lb>b.lb||(a.lb==b.lb&&(a.leaf>b.leaf||(a.leaf==b.leaf&&a.group>b.group)));}};
struct alignas(64) Counts {uint64_t scored=0,bounds=0,pairs=0,consumed=0,kept=0,enum_ids=0,max_unique=0,max_frontier=0,approx=0;};
using Seen=std::unordered_set<uint32_t>;
struct Workspace {Seen seen;std::vector<Item> exact,kept;std::vector<Task> tasks;};
static bool accept(Item c,Workspace& w,const Native& v,float inverse,Counts& count){++count.consumed;for(auto a:w.kept){++count.pairs;if(v.score(a.id,c.id)<=c.distance*inverse)return false;}w.kept.push_back(c);return true;}
static void full(uint32_t u,Workspace& w,const Native& v,const std::vector<uint64_t>& lo,const std::vector<uint32_t>& ids,const std::vector<uint64_t>& so,const std::vector<uint32_t>& sl,float inverse,Counts& count,
 bool heap=false
 ){
 w.seen.clear();w.exact.clear();w.kept.clear();
 for(size_t j=so[u];j<so[u+1];++j){uint32_t l=sl[j];for(size_t k=lo[l];k<lo[l+1];++k){auto id=ids[k];++count.enum_ids;if(id!=u&&w.seen.insert(id).second){
  w.exact.push_back({id,v.score(u,id)});++count.scored;
 }}}
 if(heap){
 // Same total (distance,ID) order, but produce only the consumed prefix.
 std::make_heap(w.exact.begin(),w.exact.end(),ItemCmp{});
 while(!w.exact.empty()&&w.kept.size()<32){std::pop_heap(w.exact.begin(),w.exact.end(),ItemCmp{});auto c=w.exact.back();w.exact.pop_back();accept(c,w,v,inverse,count);}
 }else{
 std::sort(w.exact.begin(),w.exact.end(),rank);for(auto c:w.exact){accept(c,w,v,inverse,count);if(w.kept.size()==32)break;}
 }
 count.max_unique=std::max<uint64_t>(count.max_unique,w.seen.size());count.kept+=w.kept.size();
}
static void lazy(uint32_t u,Workspace& w,const Native& v,const std::vector<Tree>& trees,const std::vector<uint32_t>& ids,const std::vector<uint64_t>& so,const std::vector<uint32_t>& sl,float inverse,Counts& count){
 w.seen.clear();w.exact.clear();w.kept.clear();w.tasks.clear();
 auto enqueue=[&](uint32_t t,uint32_t k){w.tasks.push_back({lower(u,trees[t].nodes[k],v),t,k});std::push_heap(w.tasks.begin(),w.tasks.end(),TaskCmp{});++count.bounds;};
 for(size_t j=so[u];j<so[u+1];++j)enqueue(sl[j],0);
 while(w.kept.size()<32){
  while(!w.tasks.empty()&&(w.exact.empty()||w.tasks.front().lb<=double(w.exact.front().distance))){std::pop_heap(w.tasks.begin(),w.tasks.end(),TaskCmp{});auto t=w.tasks.back();w.tasks.pop_back();auto& g=trees[t.leaf].nodes[t.group];
   if(g.left!=UINT32_MAX){enqueue(t.leaf,g.left);enqueue(t.leaf,g.right);}
   else for(size_t k=g.begin;k<g.end;++k){auto id=ids[k];++count.enum_ids;if(id!=u&&w.seen.insert(id).second){w.exact.push_back({id,v.score(u,id)});std::push_heap(w.exact.begin(),w.exact.end(),ItemCmp{});++count.scored;}}
   count.max_frontier=std::max<uint64_t>(count.max_frontier,w.tasks.size()+w.exact.size());
  }
  if(w.exact.empty())break;std::pop_heap(w.exact.begin(),w.exact.end(),ItemCmp{});auto c=w.exact.back();w.exact.pop_back();accept(c,w,v,inverse,count);
 }count.max_unique=std::max<uint64_t>(count.max_unique,w.seen.size());count.kept+=w.kept.size();
}
void build(const float* data,size_t n,size_t d,size_t pitch,const std::vector<uint64_t>& lo,std::vector<uint32_t>& ids,const std::vector<uint64_t>& so,const std::vector<uint32_t>& sl,Edge* out,uint8_t* degrees,double alpha,bool bounds,bool check){
 if(std::fegetround()!=FE_TONEAREST||d>4096||alpha!=1.2||pitch<d)throw std::runtime_error("ordered stream L2 FP32 alpha1.2 supported only");Native v{data,d,pitch};std::vector<Tree> trees;auto start=std::chrono::steady_clock::now();uint64_t tree_bytes=0,nodes=0;
 if(bounds){trees.resize(lo.size()-1);parlay::parallel_for(0,trees.size(),[&](size_t l){if(lo[l]!=lo[l+1])build_tree(trees[l],ids,lo[l],lo[l+1],v);});tree_bytes=trees.capacity()*sizeof(Tree);for(auto& t:trees){tree_bytes+=t.nodes.capacity()*sizeof(Group);nodes+=t.nodes.size();}}
 std::cout<<"ORDER_FULL_TREES bounds="<<bounds<<" payload_bytes="<<tree_bytes<<" nodes="<<nodes<<" seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<" center_coordinates_referenced=1\n"<<std::flush;
 std::vector<Counts> counters(parlay::num_workers());std::vector<Workspace> workspace(parlay::num_workers());start=std::chrono::steady_clock::now();std::atomic<size_t> checked{0};
 parlay::parallel_for(0,n,[&](size_t u){auto worker=parlay::worker_id();auto& w=workspace[worker];auto& c=counters[worker];
  if(bounds)lazy(u,w,v,trees,ids,so,sl,float(1./alpha),c);else full(u,w,v,lo,ids,so,sl,float(1./alpha),c,
   false
  );
  if(check&&u%4096==0){Workspace oracle;Counts unused;full(u,oracle,v,lo,ids,so,sl,float(1./alpha),unused);if(w.kept.size()!=oracle.kept.size())throw std::runtime_error("ordered full degree oracle");for(size_t k=0;k<w.kept.size();++k)if(w.kept[k].id!=oracle.kept[k].id||std::memcmp(&w.kept[k].distance,&oracle.kept[k].distance,4))throw std::runtime_error("ordered full FP32/ID oracle");++checked;}
  degrees[u]=w.kept.size();for(size_t k=0;k<w.kept.size();++k){Eigen::bfloat16 bf(w.kept[k].distance);uint16_t bits;std::memcpy(&bits,&bf,2);out[u*32+k]={w.kept[k].id,bits};}
 });Counts sum;uint64_t workspace_bytes=0;for(size_t i=0;i<counters.size();++i){auto c=counters[i];sum.approx+=c.approx;sum.scored+=c.scored;sum.bounds+=c.bounds;sum.pairs+=c.pairs;sum.consumed+=c.consumed;sum.kept+=c.kept;sum.enum_ids+=c.enum_ids;sum.max_unique=std::max(sum.max_unique,c.max_unique);sum.max_frontier=std::max(sum.max_frontier,c.max_frontier);auto& w=workspace[i];workspace_bytes+=w.exact.capacity()*sizeof(Item)+w.kept.capacity()*sizeof(Item)+w.tasks.capacity()*sizeof(Task);
 workspace_bytes+=w.seen.bucket_count()*sizeof(void*)+w.seen.size()*(sizeof(uint32_t)+2*sizeof(void*));
 }
 std::cout<<"ORDER_FULL_DISCOVERY bounds="<<bounds<<" seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<" unique_scores="<<sum.scored<<" lower_bounds="<<sum.bounds<<" prune_pair_distances="<<sum.pairs<<" enumerated="<<sum.enum_ids<<" consumed="<<sum.consumed<<" retained="<<sum.kept<<" max_worker_unique="<<sum.max_unique<<" max_frontier="<<sum.max_frontier<<" workspace_estimate_bytes="<<workspace_bytes<<" checked_rows="<<checked<<" global_candidate_capacity=32 global_distance_matrix=0\n"<<std::flush;
}
void unit(){
 std::mt19937 rng(20261009);for(size_t d:{size_t(96),size_t(100),size_t(128)})for(size_t kind=0;kind<5;++kind){size_t n=257;std::vector<float> x(n*d);for(auto& a:x)a=float(int(rng()%2001)-1000)/1000;if(kind==1)std::fill(x.begin(),x.end(),0);if(kind==2)for(auto& a:x)a=1+a*0x1p-22f;if(kind==3)for(auto& a:x)a*=0x1p-70f;if(kind==4)for(auto& a:x)a*=0x1p25f;
  std::vector<uint64_t> lo{0,n,n*2},so(n+1);std::vector<uint32_t> ids(n*2),sl(n*2);for(size_t k=0;k<n;++k){ids[k]=k;ids[n+k]=n-1-k;so[k]=2*k;sl[2*k]=0;sl[2*k+1]=1;}so[n]=n*2;std::vector<Tree> trees(2);Native v{x.data(),d,d};build_tree(trees[0],ids,0,n,v);build_tree(trees[1],ids,n,n*2,v);
  for(size_t u=0;u<32;++u){Workspace a,b;Counts ac,bc;full(u,a,v,lo,ids,so,sl,float(1./1.2),ac);lazy(u,b,v,trees,ids,so,sl,float(1./1.2),bc);if(a.kept.size()!=b.kept.size())throw std::runtime_error("ordered unit degree");for(size_t k=0;k<a.kept.size();++k)if(a.kept[k].id!=b.kept[k].id||std::memcmp(&a.kept[k].distance,&b.kept[k].distance,4))throw std::runtime_error("ordered unit ID/bits");
   // Independent canonical full-set/sort reference, including duplicate IDs.
   Workspace canonical;Counts cc;std::vector<Item> all;for(uint32_t id=0;id<n;++id)if(id!=u)all.push_back({id,v.score(u,id)});std::sort(all.begin(),all.end(),rank);for(auto c:all){accept(c,canonical,v,float(1./1.2),cc);if(canonical.kept.size()==32)break;}if(a.kept.size()!=canonical.kept.size()||ac.consumed!=cc.consumed||ac.pairs!=cc.pairs)throw std::runtime_error("ordered canonical sorted stream");for(size_t k=0;k<a.kept.size();++k)if(a.kept[k].id!=canonical.kept[k].id||std::memcmp(&a.kept[k].distance,&canonical.kept[k].distance,4))throw std::runtime_error("ordered canonical sorted ID/bits");
   for(auto& t:trees)for(auto& g:t.nodes)for(size_t k=g.begin;k<g.end;++k)if(lower(u,g,v)>v.score(u,ids[k]))throw std::runtime_error("ordered unit native lower bound");}
 }std::cout<<"ORDER_FULL_UNIT ties_self_duplicates_native_bounds_ordered_FP32=passed\n";
}
}

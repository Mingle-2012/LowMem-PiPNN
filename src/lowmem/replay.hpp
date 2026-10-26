#pragma once
#include "owned_pages.hpp"
#include "topology.hpp"
#include "reverse_candidates.hpp"
struct IdCoupling {
 RawTopology kernel;OwnedPages<uint32_t> slots;std::vector<uint32_t> degrees;
 // References address the immutable concatenation E(u), G0(u), cached NEW(u).
 // Final uint32 slots are allocated only after the cross-row read barrier.
 OwnedPages<uint8_t> first_positions;
 template<class E,class G> uint32_t first_id(size_t u,size_t k,const E& e,const G& g)const{
  size_t p=first_positions[u*lowmem_degree()+k],a=e.offsets[u+1]-e.offsets[u],b=g.offsets[u+1]-g.offsets[u];
  if(p<a)return e.neighbors[e.offsets[u]+p];p-=a;
  if(p<b)return g.neighbors[g.offsets[u]+p];p-=b;
  if(p>=saved_extra.counts[u])throw std::runtime_error("first reference out of range");
  return saved_extra.ids[u*64+p];
 }
 template<class E,class G> void save_first(size_t u,const List& kept,const E& e,const G& g){

  std::array<uint32_t,512> keys;keys.fill(UINT32_MAX);std::array<uint8_t,512> ordinals;size_t count=0;
  auto slot=[](uint32_t id){uint32_t x=id*2654435761u;x^=x>>16;return size_t(x&511u);};
  auto add=[&](uint32_t id){if(count==160)throw std::runtime_error("first reference input exceeds160");size_t k=slot(id);while(keys[k]!=UINT32_MAX&&keys[k]!=id)k=(k+1)&511u;if(keys[k]==UINT32_MAX){keys[k]=id;ordinals[k]=uint8_t(count);}++count;};
  for(size_t j=e.offsets[u];j<e.offsets[u+1];++j)add(e.neighbors[j]);
  for(size_t j=g.offsets[u];j<g.offsets[u+1];++j)add(g.neighbors[j]);
  for(size_t j=0;j<saved_extra.counts[u];++j)add(saved_extra.ids[u*64+j]);
  for(size_t k=0;k<kept.size();++k){size_t p=slot(kept[k].id);while(keys[p]!=UINT32_MAX&&keys[p]!=kept[k].id)p=(p+1)&511u;if(keys[p]==UINT32_MAX)throw std::runtime_error("first retained ID absent from input");first_positions[u*lowmem_degree()+k]=ordinals[p];if(first_id(u,k,e,g)!=kept[k].id)throw std::runtime_error("first reference ordered roundtrip mismatch");}

 }
 uint64_t reverse_count=0,capped=0,capped_rows=0,replayed_rows=0,full_second_rows=0,rejected_records=0;
 brev::Index g0_reverse;
 brev::SelectedExtras saved_extra;
 static constexpr bool compact_reject=false;
 OwnedPages<uint32_t> rejected_ids;std::vector<uint32_t> rejected_degrees;
 IdCoupling(const float* x,size_t n,size_t d,size_t T):kernel(x,n,d,T,false),slots(
 0
 ),degrees(n)
 ,saved_extra(
 n
 )
 {
 static_assert(!compact_reject,"position references require cached normal Replay");
 if(lowmem_degree()!=32)throw std::runtime_error("first-position ablation requires R32");
 first_positions=OwnedPages<uint8_t>(n*lowmem_degree());
 std::cout<<"FIRST_POSITION refs_bytes="<<first_positions.bytes<<" degrees_bytes="<<degrees.capacity()*4<<" final_ID_bits=32 input_cap=160 cone_EF=0"<<std::endl;
 if(compact_reject){rejected_ids=OwnedPages<uint32_t>(n*128);rejected_degrees.resize(n);}
 }
 template<class E,class G> void pool(size_t u,const E& evidence,const G& g,List& out)const{
  brev::pool(kernel,u,evidence,g,g0_reverse,out);return;
 }
 template<class E,class G> void first(const E& evidence,const G& g){
  if(brev::post_mode!=2||brev::depth!=1||64!=64)throw std::runtime_error("shared-target ablation requires reverse1/shared NEW cap64 without forward twohop");
  std::cout<<"SHARED_TARGET_POLICY relation=out_then_in retained_base=E_union_G0 extra_cap=64 forward_twohop=0 anchor_pair_scatter=0"<<std::endl;
  if(brev::post_mode&2)g0_reverse.build(g,kernel.threads);
  lowmem_profile::mark("B_pass_A");
  if(compact_reject)throw std::runtime_error("selected extra cache requires normal Replay");
  saved_extra.prepare(evidence,g,g0_reverse,kernel.threads);
  std::cout<<"SHARED_TOP64_CACHE id_bytes="<<saved_extra.ids.bytes<<" count_bytes="<<saved_extra.counts.capacity()<<" scores_stored=0 replay_discovery=0 replay_native_rescore=base_plus_selected release_before_final_encoding=1"<<std::endl;
#pragma omp parallel num_threads(kernel.threads)
  {List candidates,kept,rejected;
#pragma omp for schedule(dynamic)
   for(size_t u=0;u<kernel.n;++u){
    pool(u,evidence,g,candidates);
    saved_extra.save(u,brev::workspace.extra);
    kept.clear();rejected.clear();{
    kernel.prune(candidates,kept,rejected,u);}
    degrees[u]=kept.size();
    save_first(u,kept,evidence,g);
    if(compact_reject){if(rejected.size()>128)throw std::runtime_error("first rejected capacity exceeded");rejected_degrees[u]=rejected.size();for(size_t k=0;k<rejected.size();++k)rejected_ids[u*128+k]=rejected[k].id;}}
   }
  // SelectedExtras owns copied global IDs; all future Replay calls use it.
  // The first-pass OpenMP region has joined. first_id also uses only E/G0/NEW.
  std::cout<<"PHASE name=first_pass_ready rss="<<lowmem::memory_rss()<<std::endl;
  size_t released_reverse=g0_reverse.ids.bytes+g0_reverse.offsets.capacity()*sizeof(size_t);
  g0_reverse.ids.reset();std::vector<size_t>().swap(g0_reverse.offsets);
  std::cout<<"PHASE name=first_discovery_released rss="<<lowmem::memory_rss()<<std::endl;
  std::cout<<"G0_DISCOVERY_RELEASE bytes="<<released_reverse<<" after_first_pass_barrier=1 selected_global_ID_cache_preserved=1 future_replay_uses_saved_extra=1"<<std::endl;
  if(compact_reject)for(auto count:rejected_degrees)rejected_records+=count;
 }
 template<class E,class G> uint32_t retained_id(size_t u,size_t k,const E& e,const G& g)const{
  return first_id(u,k,e,g);
 }
 template<class E,class G> bool reciprocal(uint32_t u,uint32_t v,const E& e,const G& g)const{for(size_t k=0;k<degrees[v];++k)if(retained_id(v,k,e,g)==u)return true;return false;}
 template<class E,class G> void reverse_and_finish(const E& evidence,const G& g){
  lowmem_profile::mark("reverse_count_scatter");const size_t n=kernel.n,T=kernel.threads;
  std::cout<<"NO_FINAL_REVERSE discovery_reverse_retained=1 final_proposals=0 second_prune=0 Replay_and_completion_retained=1"<<std::endl;
  lowmem_profile::mark("reverse_barrier_row_completion");
  // All cross-row reads of FIRST-PASS K end at this barrier. Thereafter each
  // worker reads/writes only its own K slot, immutable evidence/reverse and X.
  slots=OwnedPages<uint32_t>(n*lowmem_degree());
  const size_t rows_per_page=4096/lowmem_degree(),page_count=(n+rows_per_page-1)/rows_per_page;
  auto pending=std::make_unique<std::atomic<uint32_t>[]>(page_count);
  for(size_t p=0;p<page_count;++p)pending[p].store(std::min(rows_per_page,n-p*rows_per_page));
  std::atomic<bool> failed{false};std::exception_ptr failure;uint64_t trimmed=0,trimmed_rows=0,full_rows=0;
#pragma omp parallel num_threads(T) reduction(+:trimmed,trimmed_rows,full_rows)
  {List candidates,first,rejected,second,pair_pool;
#pragma omp for schedule(dynamic)
   for(size_t u=0;u<n;++u){if(failed.load(std::memory_order_relaxed))continue;try{
    first.clear();rejected.clear();
    if(compact_reject){
     for(size_t k=0;k<degrees[u];++k){auto v=slots[u*lowmem_degree()+k];first.emplace_back(v,kernel.distance(u,v));}
     for(size_t k=0;k<rejected_degrees[u];++k){auto v=rejected_ids[u*128+k];rejected.emplace_back(v,kernel.distance(u,v));}
    }else{
     saved_extra.pool(kernel,u,evidence,g,candidates);
     {
     kernel.prune(candidates,first,rejected,u);}
     if(first.size()!=degrees[u])throw std::runtime_error("first-pass replay degree mismatch");for(size_t k=0;k<first.size();++k)if(first[k].id!=retained_id(u,k,evidence,g))throw std::runtime_error("first-pass replay ordered ID mismatch");
    }
    second=first;
    full_rows+=second.size()==lowmem_degree();{
    kernel.complete(u,second,rejected);}
    if(second.size()!=lowmem_degree())throw std::runtime_error("completion degree mismatch");for(size_t k=0;k<lowmem_degree();++k)slots[u*lowmem_degree()+k]=second[k].id;degrees[u]=lowmem_degree();
    size_t page=u/rows_per_page;if(pending[page].fetch_sub(1,std::memory_order_acq_rel)==1)first_positions.retire(page*rows_per_page*lowmem_degree(),std::min(n,(page+1)*rows_per_page)*lowmem_degree());
   }catch(...){
#pragma omp critical(b_finish_failure)
    {if(!failure)failure=std::current_exception();}failed.store(true,std::memory_order_relaxed);}}
   bcost_clear_current();
  }
  if(failure)std::rethrow_exception(failure);replayed_rows=compact_reject?0:n;
  first_positions.reset();std::cout<<"FIRST_POSITION_RELEASE ordered_Replay_checks="<<n<<" first_refs_bytes=0 global_ID_bits=32"<<std::endl;
  capped=trimmed;capped_rows=trimmed_rows;full_second_rows=full_rows;rejected_ids.reset();std::vector<uint32_t>().swap(rejected_degrees);
  saved_extra.release();std::cout<<"SHARED_TOP64_CACHE_RELEASE cached_payload_bytes=0 replay_ordered_first_rows_checked="<<replayed_rows<<std::endl;
 }
 static void bcost_clear_current(){
 }
};

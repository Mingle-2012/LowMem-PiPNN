#pragma once
#include "b64_adapter.hpp"
#include <array>
struct IdRow {
 const uint32_t* data=nullptr;size_t count=0;
 const uint32_t* begin()const{return data;}const uint32_t* end()const{return data+count;}
 size_t size()const{return count;}uint32_t operator[](size_t k)const{return data[k];}
};
inline size_t graph_nodes(const B64Seed& g){return g.offsets.size()-1;}
template<class Graph> size_t graph_nodes(const Graph& g){return g.n();}
inline IdRow graph_row(const B64Seed& g,size_t u,std::array<uint32_t,64>&){return {g.neighbors.data()+g.offsets[u],g.offsets[u+1]-g.offsets[u]};}
template<class Graph> IdRow graph_row(const Graph& g,size_t u,std::array<uint32_t,64>& scratch){auto r=g.row(u,scratch);return {r.begin(),r.size()};}

#pragma once
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <time.h>
#include "../../../parlay/include/parlay/parallel.h"
namespace parlayANN {
// Passive FP32 oracle. No estimator executes on the construction path.
struct __attribute__((packed)) PrecisionTraceEvent {
 uint64_t stamp;uint32_t u,v,boundary;float x,t;uint8_t cls,flags,result,reserved;
};
static_assert(sizeof(PrecisionTraceEvent)==32,"trace layout");
struct DecisionPrecisionDiagnostic {
  struct alignas(64) Worker { FILE* file=nullptr; uint64_t counts[11][2]{}; };
  std::array<Worker,256> workers{};
  std::string root;
  bool full=false;
  DecisionPrecisionDiagnostic() { const char* p=std::getenv("PIPNN_PRECISION_TRACE"); if(p) root=p; full=std::getenv("PIPNN_PRECISION_FULL_TRACE")!=nullptr; }
  ~DecisionPrecisionDiagnostic() {
    if(root.empty()) return;
    uint64_t sums[11][2]{};
    for(auto& w:workers) { if(w.file) std::fclose(w.file); for(int c=0;c<11;++c)for(int b=0;b<2;++b)sums[c][b]+=w.counts[c][b]; }
    FILE* f=std::fopen((root+"/counts.csv").c_str(),"w"); if(!f) std::abort();
    std::fprintf(f,"class,false,true\n");for(int c=0;c<11;++c)std::fprintf(f,"%d,%llu,%llu\n",c,(unsigned long long)sums[c][0],(unsigned long long)sums[c][1]);std::fclose(f);
  }
  void record(int cls,uint32_t source,uint32_t u,uint32_t v,uint32_t boundary,float x,float t,bool result,int flags=0) {
    if(root.empty())return;
    auto wid=parlay::worker_id();if(wid>=workers.size())std::abort();auto& w=workers[wid];++w.counts[cls][result];
    if(full) {
      // Stored scalar scans/outcomes have counts but require no vector distances.
      if(cls==3 || cls==5 || cls==7)return;
      if(!w.file){w.file=std::fopen((root+"/full_"+std::to_string(wid)+".bin").c_str(),"wb");if(!w.file)std::abort();std::setvbuf(w.file,nullptr,_IOFBF,8<<20);}
      timespec now;clock_gettime(CLOCK_MONOTONIC_RAW,&now);
      PrecisionTraceEvent e{uint64_t(now.tv_sec)*1000000000ull+uint64_t(now.tv_nsec),u,v,boundary,x,t,uint8_t(cls),uint8_t(flags | ((cls==6 && x==t && result)?128:0)),uint8_t(result),0};
      if(std::fwrite(&e,sizeof(e),1,w.file)!=1)std::abort();
      return;
    }
    // Fixed source sample, independent of distance/outcome/thread scheduling.
    if(((uint64_t(source)*11400714819323198485ull)>>54)!=0)return;
    if(!w.file){w.file=std::fopen((root+"/trace_"+std::to_string(wid)+".csv").c_str(),"w");if(!w.file)std::abort();std::setvbuf(w.file,nullptr,_IOFBF,1<<20);}
    std::fprintf(w.file,"%d,%u,%u,%u,%u,%.9g,%.9g,%d,%d\n",cls,source,u,v,boundary,x,t,int(result),flags);
  }
};
inline DecisionPrecisionDiagnostic& precision_diagnostic(){static DecisionPrecisionDiagnostic d;return d;}
}
#define PIPNN_PRECISION_RECORD(...) ((void)0)

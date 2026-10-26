#pragma once
#include <thread>
#include <mutex>
#include <atomic>
#include <fstream>
#include <chrono>
#include <sys/resource.h>
#include <cstdlib>
#include <array>
#include <iostream>
#include <ctime>
#include <omp.h>
namespace lowmem_profile {
using Clock=std::chrono::steady_clock;
inline Clock::time_point origin;
inline std::mutex lock;
inline std::string phase="not_started";
inline std::ofstream events;
inline double cpu(){rusage r{};getrusage(RUSAGE_SELF,&r);return r.ru_utime.tv_sec+r.ru_utime.tv_usec/1e6+r.ru_stime.tv_sec+r.ru_stime.tv_usec/1e6;}
inline void mark(const char* p){std::lock_guard<std::mutex> l(lock);phase=p;if(events)events<<std::chrono::duration<double>(Clock::now()-origin).count()<<","<<cpu()<<","<<p<<"\n"<<std::flush;}
enum Function {Find,Augment,Prune1,Prune2,Complete,Update,Count};
struct Cost {double cpu=0,wall=0;uint64_t calls=0;};
inline std::array<std::array<Cost,Count>,1024> costs{};
inline double thread_cpu(){timespec t{};clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t);return t.tv_sec+t.tv_nsec/1e9;}
struct Scope{
 bool enabled;Function f;double start_cpu=0;Clock::time_point start_wall;
 Scope(Function function,bool active):enabled(active),f(function){if(enabled){start_cpu=thread_cpu();start_wall=Clock::now();}}
 ~Scope(){if(enabled){auto& c=costs.at(omp_get_thread_num())[f];c.cpu+=thread_cpu()-start_cpu;c.wall+=std::chrono::duration<double>(Clock::now()-start_wall).count();++c.calls;}}
};
inline void reset_costs(){for(auto& row:costs)row={};}
inline void print_costs(){const char* names[]={"find_candidates","augmentation_sort","prune1","prune2","complete","update_qg_final"};for(int f=0;f<Count;++f){Cost sum;for(const auto& row:costs){sum.cpu+=row[f].cpu;sum.wall+=row[f].wall;sum.calls+=row[f].calls;}std::cout<<"FUNCTION name="<<names[f]<<" cpu_seconds="<<sum.cpu<<" summed_row_wall_seconds="<<sum.wall<<" calls="<<sum.calls<<std::endl;}}
struct Sampler{
 std::atomic<bool> done{false};std::thread worker;bool active=false;
 Sampler(){auto path=std::getenv("LOWMEM_TIMELINE");origin=Clock::now();if(!path)return;active=true;events.open(std::string(path)+".events.csv");events<<"seconds,cpu_seconds,phase\n";
 worker=std::thread([this,path=std::string(path)]{std::ofstream out(path);out<<"seconds,phase,VmRSS_bytes,VmHWM_bytes,AnonHugePages_bytes\n";while(!done){size_t rss=0,hwm=0,huge=0;std::string k,line;size_t v;std::ifstream f("/proc/self/status");while(f>>k){if(k=="VmRSS:"||k=="VmHWM:"){f>>v;if(k=="VmRSS:")rss=v*1024;else hwm=v*1024;}std::getline(f,line);}std::ifstream s("/proc/self/smaps_rollup");while(s>>k){if(k=="AnonHugePages:"){s>>v;huge=v*1024;}std::getline(s,line);}std::string current;{std::lock_guard<std::mutex> l(lock);current=phase;}out<<std::chrono::duration<double>(Clock::now()-origin).count()<<","<<current<<","<<rss<<","<<hwm<<","<<huge<<"\n"<<std::flush;std::this_thread::sleep_for(std::chrono::milliseconds(100));}});
 }
 void stop(){if(active){done=true;worker.join();active=false;events.close();}}
 ~Sampler(){stop();}
};
}

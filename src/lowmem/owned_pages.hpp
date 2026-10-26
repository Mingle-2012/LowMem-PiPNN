#pragma once
#include <sys/mman.h>
#include <unistd.h>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <algorithm>
// Ordinary owned DRAM allocation. Dead complete pages may be relinquished only
// after their last read; never files, checkpoints or an external memory store.
template<class T> struct OwnedPages {
 T* pointer=nullptr;size_t count=0,bytes=0;
 OwnedPages()=default;explicit OwnedPages(size_t n):count(n){bytes=((n*sizeof(T)+4095)/4096)*4096;if(posix_memalign(reinterpret_cast<void**>(&pointer),4096,bytes))throw std::bad_alloc();for(size_t i=0;i<n;++i)::new(pointer+i)T;}
 ~OwnedPages(){reset();}OwnedPages(const OwnedPages&)=delete;OwnedPages& operator=(const OwnedPages&)=delete;
 OwnedPages(OwnedPages&& o)noexcept:pointer(o.pointer),count(o.count),bytes(o.bytes){o.pointer=nullptr;o.count=o.bytes=0;}
 OwnedPages& operator=(OwnedPages&& o)noexcept{reset();pointer=o.pointer;count=o.count;bytes=o.bytes;o.pointer=nullptr;o.count=o.bytes=0;return *this;}
 void reset(){std::free(pointer);pointer=nullptr;count=bytes=0;}
 T* data(){return pointer;}const T* data()const{return pointer;}
 T& operator[](size_t i){return pointer[i];}const T& operator[](size_t i)const{return pointer[i];}
 size_t retire(size_t first,size_t last){size_t a=first*sizeof(T),b=last*sizeof(T);a=(a+4095)/4096*4096;b=b/4096*4096;if(b<=a)return 0;if(madvise(reinterpret_cast<char*>(pointer)+a,b-a,MADV_DONTNEED))throw std::runtime_error("dead page retirement failed");return b-a;}
};

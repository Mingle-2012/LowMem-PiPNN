#pragma once
// Construction-only MIPS policy. Scores passed here are native -dot, never
// squared L2 or 1-dot. No L2 alpha is accepted by this interface.
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>
namespace mips_prune {
enum class Strategy { AngularNorm, Napg };
// Isolated ablation only; ordinary main builds retain the strict alpha-free
// predicate. This is an IP similarity multiplier, not the L2 prune alpha.
inline constexpr float napg_alpha(){
  return 1.0F;
}
inline Strategy strategy() {
  static const Strategy value=[] {
    const char* p=std::getenv("LOWMEM_MIPS_PRUNE");
    const std::string s=p?p:"napg";
    if(s=="angular_norm")return Strategy::AngularNorm;
    if(s=="napg")return Strategy::Napg;
    throw std::runtime_error("LOWMEM_MIPS_PRUNE must be angular_norm or napg");
  }();
  return value;
}
inline const char* name(){return strategy()==Strategy::Napg?(napg_alpha()==1.F?"napg_strict":"napg_alpha12"):"angular_norm_two_witness";}
template<class Selected,class Score,class Norm>
bool occluded(unsigned u,unsigned v,float source_score,size_t count,
              Selected selected,Score score,Norm norm,Strategy policy) {
  if(policy==Strategy::Napg) {
    // Strict w.dot(v) > alpha_IP*u.dot(v), equivalently -dot(w,v) <
    // alpha_IP*(-dot(u,v)). Never multiply RAW's shifted 1-dot score.
    const float threshold=source_score*napg_alpha();
    for(size_t j=0;j<count;++j)if(score(selected(j),v)<threshold)return true;
    return false;
  }
  const double nu=norm(u),nv=norm(v);
  if(!(nu>0&&nv>0))return false;
  const double uv=-double(source_score)/(nu*nv);
  unsigned witnesses=0;
  for(size_t j=0;j<count;++j) {
    const unsigned w=selected(j);const double nw=norm(w);
    if(!(nw>0)||nw<nv)continue;
    const double uw=-double(score(u,w))/(nu*nw);
    if(!(uw>uv))continue;
    const double wv=-double(score(w,v))/(nw*nv);
    if(wv>uv&&++witnesses==2)return true;
  }
  return false;
}
}

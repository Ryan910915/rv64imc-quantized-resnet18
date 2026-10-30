// Measurement-only observer for Spike. No guest instruction semantics change.
// Counts [opening minstret read, closing minstret read), matching after-before.
// Call ONLY after successful execution and when npc != PC_SERIALIZE_BEFORE.
#ifndef RESNET_PROFILE_MIX_OBSERVER_H
#define RESNET_PROFILE_MIX_OBSERVER_H
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <limits>

namespace resnet_profile_mix {
struct Slot { uint64_t bits=0, count=0; unsigned length=0; };
class Observer {
 public:
  bool enabled=false;
  Observer() {
    const char *pc=std::getenv("RESNET_MIX_COUNTER_PC");
    if (!pc) return;
    gate=number(pc); base=number(required("RESNET_MIX_TEXT_BASE"));
    limit=number(required("RESNET_MIX_TEXT_LIMIT"));
    expected=number(required("RESNET_MIX_REGIONS"));
    if ((base&1) || limit<=base || limit-base>UINT64_C(16777216) ||
        gate<base || gate>=limit || !expected)
      throw std::runtime_error("Invalid RESNET_MIX configuration");
    slots.resize(static_cast<size_t>((limit-base+1)/2));
    path=required("RESNET_MIX_OUTPUT");
    // The driver uses a unique output directory; never silently truncate evidence.
    file=std::fopen(path.c_str(),"wx");
    if (!file) throw std::runtime_error("Cannot create profile mix output: "+path);
    std::fprintf(file,"# RESNET_MIX_V1 gate=0x%llx expected=%llu\n",
                 (unsigned long long)gate,(unsigned long long)expected);
    std::fprintf(file,"op,pc,bits,length,count\n");
    enabled=true;
  }
  ~Observer() {
    if (!file) return;
    std::fprintf(file,"# END regions=%llu expected=%llu active=%u\n",
                 (unsigned long long)completed,(unsigned long long)expected,active?1:0);
    std::fclose(file);
  }
  void retired(uint64_t hart, uint64_t pc,uint64_t bits,unsigned length) {
    if (!enabled) return;
    // Spike can sign-extend fetched encodings; normalize to instruction width.
    if (length==0 || length>8) throw std::runtime_error("Invalid instruction length");
    if (length<8) bits &= (UINT64_C(1)<<(8*length))-1;
    if (hart!=0) throw std::runtime_error("Profile mix supports one hart (hart 0) only");
    if (pc==gate) {
      if (length!=4 || uint32_t(bits)!=UINT32_C(0xb0202573))
        throw std::runtime_error("Gate is not csrrs a0,minstret,x0");
      if (active) { finish(); active=false; return; }
      if (completed>=expected) throw std::runtime_error("Unexpected extra counter pair");
      active=true; total=0;
    }
    if (!active) return;
    if (pc<base || pc>=limit || (pc&1))
      throw std::runtime_error("Retired PC outside configured executable span in ROI");
    Slot &s=slots[static_cast<size_t>((pc-base)/2)];
    if (s.length && (s.bits!=bits || s.length!=length))
      throw std::runtime_error("Self-modifying instruction detected; unsupported");
    s.bits=bits;s.length=length;
    if (s.count==std::numeric_limits<uint64_t>::max() ||
        total==std::numeric_limits<uint64_t>::max())
      throw std::runtime_error("Instruction counter overflow");
    ++s.count;++total;
  }
 private:
  uint64_t gate=0,base=0,limit=0,expected=0,completed=0,total=0;
  bool active=false;FILE *file=nullptr;std::string path;std::vector<Slot> slots;
  static const char *required(const char *key) {
    const char *s=std::getenv(key);
    if (!s || !*s) throw std::runtime_error(std::string("Missing ")+key);
    return s;
  }
  static uint64_t number(const char *s) {
    char *end=nullptr;errno=0;
    if (!s || !*s || *s=='-') throw std::runtime_error("Invalid unsigned number");
    unsigned long long x=std::strtoull(s,&end,0);
    if (errno || !end || *end) throw std::runtime_error("Invalid unsigned number");
    return static_cast<uint64_t>(x);
  }
  void finish() {
    for (size_t i=0;i<slots.size();++i) {
      Slot &s=slots[i];if (!s.count) continue;
      std::fprintf(file,"%llu,0x%llx,0x%llx,%u,%llu\n",
          (unsigned long long)completed,(unsigned long long)(base+2*i),
          (unsigned long long)s.bits,s.length,(unsigned long long)s.count);
      s.count=0;
    }
    std::fprintf(file,"# CLOSE op=%llu total=%llu\n",
                 (unsigned long long)completed,(unsigned long long)total);
    if (std::fflush(file) || std::ferror(file))
      throw std::runtime_error("Cannot write profile mix output");
    ++completed;
  }
};
inline Observer &instance() {static Observer observer;return observer;}
} // namespace resnet_profile_mix
#endif

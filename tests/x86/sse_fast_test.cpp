// The host fast paths (sse_fast.h; CvtIntToF's in sse.cpp) of Arith,
// Sqrt, MinMax, CvtFloat and CvtIntToF against the exact SoftFloat paths: random add/sub/mul/div of float and double lanes, biased
// towards the edges of the fast path (near overflow and underflow, exact
// cancellation, zeros, denormals, NaNs and infinities, sticky flags already
// set), must give the same result bits and the same MXCSR.

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#include "x86/sse.h"

namespace recomp {
void FatalF(const char* fmt, ...) {  // x86/sse.cpp's #XM
  va_list ap;
  va_start(ap, fmt);
  std::vprintf(fmt, ap);
  va_end(ap);
  std::abort();
}
}  // namespace recomp

namespace {

std::mt19937_64 g_rng(0x5EEDF00Dull);

uint64_t Rand() { return g_rng(); }

// A biased random binary32 or binary64 bit pattern.
template <bool D>
uint64_t Operand() {
  const int ebits = D ? 11 : 8, fbits = D ? 52 : 23;
  const uint64_t emax = (1ull << ebits) - 1;
  uint64_t sign = Rand() & 1, frac = Rand() & ((1ull << fbits) - 1), exp;
  switch (Rand() % 10) {
    case 0: exp = 0; if (Rand() & 1) frac = 0; break;         // zero or denormal
    case 1: exp = emax; break;                                // inf or NaN
    case 2: exp = 1 + Rand() % 4; break;                      // just above underflow
    case 3: exp = emax - 1 - Rand() % 4; break;               // just below overflow
    case 4: frac = 0; exp = Rand() % emax; break;             // powers of two, zero
    case 5: frac &= ~((1ull << (fbits - 4)) - 1); exp = (emax >> 1) + Rand() % 8; break;
    default: exp = (emax >> 1) - 40 + Rand() % 80; break;     // ordinary
  }
  return sign << (ebits + fbits) | exp << fbits | frac;
}

template <bool D>
void SetLanes(recomp::Xmm& x, const uint64_t* v, int lanes) {
  for (int i = 0; i < lanes; i++) {
    if (D)
      recomp::sse::SetLane<uint64_t>(x, i, v[i]);
    else
      recomp::sse::SetLane<uint32_t>(x, i, uint32_t(v[i]));
  }
}

int g_fail = 0;

template <bool D>
void Check(recomp::sse::Op op, int lanes) {
  uint64_t a[4], b[4];
  for (int i = 0; i < lanes; i++) {
    a[i] = Operand<D>();
    // Exact cancellation and equal magnitudes now and then.
    b[i] = Rand() % 8 == 0 ? (a[i] ^ (D ? 0x8000000000000000ull : 0x80000000ull)) : Operand<D>();
  }
  recomp::Xmm d0{}, s{};
  SetLanes<D>(d0, a, lanes);
  SetLanes<D>(s, b, lanes);
  const uint32_t sticky = uint32_t(Rand() % 4 == 0 ? Rand() & 0x3F : 0);
  recomp::Ctx c1{}, c2{};
  c1.mxcsr = c2.mxcsr = 0x1F80 | sticky;
  recomp::Xmm d1 = d0, d2 = d0;
  recomp::sse::SetFastPath(true);
  recomp::sse::Arith(c1, op, D, lanes, d1, s, 0);
  recomp::sse::SetFastPath(false);
  recomp::sse::Arith(c2, op, D, lanes, d2, s, 0);
  if (std::memcmp(&d1, &d2, sizeof d1) != 0 || c1.mxcsr != c2.mxcsr) {
    if (++g_fail <= 20)
      std::printf("FAIL %s op %d lanes %d: a0 %llx b0 %llx: mxcsr fast %x exact %x\n",
                  D ? "double" : "float", int(op), lanes, (unsigned long long)a[0],
                  (unsigned long long)b[0], c1.mxcsr, c2.mxcsr);
  }
}

template <bool D>
void CheckUnary(int kind, int lanes) {  // kind: 0 sqrt, 1 min, 2 max
  uint64_t a[4], b[4];
  for (int i = 0; i < lanes; i++) {
    a[i] = Operand<D>();
    // Equal values, and the same magnitude with the other sign (+0/-0).
    const uint64_t sign = D ? 0x8000000000000000ull : 0x80000000ull;
    b[i] = Rand() % 6 == 0 ? a[i] : Rand() % 6 == 0 ? a[i] ^ sign : Operand<D>();
    // Square roots of mostly positive values.
    if (kind == 0 && Rand() % 4) a[i] &= D ? 0x7FFFFFFFFFFFFFFFull : 0x7FFFFFFFull;
  }
  recomp::Xmm d0{}, s{};
  SetLanes<D>(d0, a, lanes);
  SetLanes<D>(s, kind == 0 ? a : b, lanes);
  recomp::Ctx c1{}, c2{};
  c1.mxcsr = c2.mxcsr = 0x1F80;
  recomp::Xmm d1 = d0, d2 = d0;
  for (int pass = 0; pass < 2; pass++) {
    recomp::sse::SetFastPath(pass == 0);
    recomp::Ctx& c = pass == 0 ? c1 : c2;
    recomp::Xmm& d = pass == 0 ? d1 : d2;
    if (kind == 0)
      recomp::sse::Sqrt(c, D, lanes, d, s, 0);
    else
      recomp::sse::MinMax(c, kind == 2, D, lanes, d, s, 0);
  }
  if ((std::memcmp(&d1, &d2, sizeof d1) != 0 || c1.mxcsr != c2.mxcsr) && ++g_fail <= 20)
    std::printf("FAIL %s kind %d lanes %d: a0 %llx b0 %llx mxcsr fast %x exact %x\n",
                D ? "double" : "float", kind, lanes, (unsigned long long)a[0],
                (unsigned long long)b[0], c1.mxcsr, c2.mxcsr);
}

void CheckCvt(bool to_dbl, int lanes) {
  recomp::Xmm s{}, d0{};
  for (int i = 0; i < lanes; i++) {
    if (to_dbl)
      recomp::sse::SetLane<uint32_t>(s, i, uint32_t(Operand<false>()));
    else
      recomp::sse::SetLane<uint64_t>(s, i, Operand<true>());
  }
  for (int i = 0; i < 2; i++) recomp::sse::SetLane<uint64_t>(d0, i, Rand());
  recomp::Ctx c1{}, c2{};
  c1.mxcsr = c2.mxcsr = 0x1F80;
  recomp::Xmm d1 = d0, d2 = d0;
  recomp::sse::SetFastPath(true);
  recomp::sse::CvtFloat(c1, to_dbl, lanes, d1, s, 0);
  recomp::sse::SetFastPath(false);
  recomp::sse::CvtFloat(c2, to_dbl, lanes, d2, s, 0);
  if ((std::memcmp(&d1, &d2, sizeof d1) != 0 || c1.mxcsr != c2.mxcsr) && ++g_fail <= 20)
    std::printf("FAIL cvt to_dbl %d lanes %d: mxcsr fast %x exact %x\n", to_dbl, lanes,
                c1.mxcsr, c2.mxcsr);
}

void CheckIntCvt(bool dbl) {
  // Mostly values that need rounding to float (|v| > 2^24), some small.
  uint64_t v = Rand() % 3 == 0 ? Rand() % 100000 : Rand();
  recomp::Xmm d1{}, d2{};
  recomp::Ctx c1{}, c2{};
  c1.mxcsr = c2.mxcsr = 0x1F80;
  recomp::sse::SetFastPath(true);
  recomp::sse::CvtIntToF(c1, dbl, false, v, d1, 0);
  recomp::sse::SetFastPath(false);
  recomp::sse::CvtIntToF(c2, dbl, false, v, d2, 0);
  if ((std::memcmp(&d1, &d2, sizeof d1) != 0 || c1.mxcsr != c2.mxcsr) && ++g_fail <= 20)
    std::printf("FAIL int cvt dbl %d v %llx: mxcsr fast %x exact %x\n", dbl,
                (unsigned long long)v, c1.mxcsr, c2.mxcsr);
}

// RSQRTPS/RCPPS lanes: every significand at two exponents (and both signs
// for rcp), then random patterns, fast against exact.
void CheckRecip() {
  auto one = [](uint32_t x) {
    if (recomp::sse::RsqrtLaneFast(x) != recomp::sse::RsqrtLane(x) && ++g_fail <= 20)
      std::printf("FAIL rsqrt %08x: fast %08x exact %08x\n", x, recomp::sse::RsqrtLaneFast(x),
                  recomp::sse::RsqrtLane(x));
    if (recomp::sse::RcpLaneFast(x) != recomp::sse::RcpLane(x) && ++g_fail <= 20)
      std::printf("FAIL rcp %08x: fast %08x exact %08x\n", x, recomp::sse::RcpLaneFast(x),
                  recomp::sse::RcpLane(x));
  };
  for (uint32_t e : {127u, 128u})
    for (uint32_t f = 0; f < 0x800000u; f++) one(e << 23 | f);
  for (uint32_t f = 0; f < 0x800000u; f += 7) one(0x80000000u | 127u << 23 | f);
  for (int i = 0; i < 4000000; i++) one(uint32_t(Rand()));
}

}  // namespace

int main() {
  CheckRecip();
  const int n = 2000000;
  for (int i = 0; i < n; i++) {
    auto op = recomp::sse::Op(i % 4);
    Check<false>(op, i % 3 == 0 ? 4 : 1);
    Check<true>(op, i % 3 == 0 ? 2 : 1);
    CheckCvt(i & 1, i % 3 == 0 ? 2 : 1);
    CheckUnary<false>(i % 3, i % 3 == 0 ? 4 : 1);
    CheckUnary<true>(i % 3, i % 3 == 0 ? 2 : 1);
    CheckIntCvt(i & 1);
  }
  if (g_fail) {
    std::printf("%d failures\n", g_fail);
    return 1;
  }
  std::printf("sse_fast: ok (%d cases per format)\n", n);
  return 0;
}

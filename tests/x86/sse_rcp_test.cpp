// sse::RcpLane (RCPPS) is the correctly rounded 1/x (see sse.h).
//
// - Every normal significand at one exponent (1 <= x < 2; the rest follows
//   from rcp(2x) = rcp(x) / 2 away from the flush threshold): the result is
//   checked against the two midpoints around it with exact integer
//   arithmetic, independent of SoftFloat.
// - The architectural error bound, |rel. error| <= 1.5 * 2^-12.
// - Special operands: ±0, denormals, ±inf, NaNs, and results that flush.

#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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

int g_fail = 0;

void Fail(const char* what, uint32_t x, uint32_t got) {
  if (++g_fail <= 20) std::printf("FAIL %s: x=%08X got %08X\n", what, x, got);
}

float F(uint32_t b) {
  float f;
  std::memcpy(&f, &b, 4);
  return f;
}

}  // namespace

int main() {
  using recomp::sse::RcpLane;
  // 1 <= x < 2 (exponent 127): x = mx * 2^-23, r = mr * 2^(er-150).
  for (uint32_t frac = 0; frac < 0x800000u; frac++) {
    uint32_t x = 0x3F800000u | frac;
    uint32_t r = RcpLane(x);
    uint64_t mx = 0x800000u | frac;
    uint32_t er = (r >> 23) & 0xFF;
    uint64_t mr = (r & 0x7FFFFF) | 0x800000u;
    // r * x == mr * mx * 2^(er - 150 - 23); 1 == 2^0. Midpoints: (2mr ± 1)
    // (the lower one 2mr - 1/2 at a power of two) in units of 2^(er-151).
    int shift = -(int(er) - 151 - 23);  // 1 in units of 2^(er-151-23)
    unsigned __int128 one = static_cast<unsigned __int128>(1) << shift;
    unsigned __int128 lo2 = 4 * mr - (mr == 0x800000u ? 1 : 2), hi2 = 4 * mr + 2;  // x2 units
    if (lo2 * mx > 2 * one || hi2 * mx < 2 * one) Fail("rounding", x, r);
    double rel = std::fabs(double(F(r)) * double(F(x)) - 1.0);
    if (rel > 1.5 / 4096) Fail("bound", x, r);
  }
  // Specials.
  struct {
    uint32_t x, want;
  } cases[] = {
      {0x00000000u, 0x7F800000u}, {0x80000000u, 0xFF800000u},  // ±0 -> ±inf
      {0x00000001u, 0x7F800000u}, {0x807FFFFFu, 0xFF800000u},  // denormal -> ±inf
      {0x7F800000u, 0x00000000u}, {0xFF800000u, 0x80000000u},  // ±inf -> ±0
      {0x7F800001u, 0x7FC00001u}, {0xFFC00000u, 0xFFC00000u},  // NaN quieted
      {0x7F7FFFFFu, 0x00000000u}, {0xFF000000u, 0x80000000u},  // tiny result: flush
      {0x3F800000u, 0x3F800000u}, {0x40000000u, 0x3F000000u},  // 1, 2
      {0xC0800000u, 0xBE800000u},                              // -4
  };
  for (auto t : cases)
    if (RcpLane(t.x) != t.want) Fail("special", t.x, RcpLane(t.x));
  if (g_fail) {
    std::printf("%d failures\n", g_fail);
    return 1;
  }
  std::printf("sse_rcp: ok\n");
  return 0;
}

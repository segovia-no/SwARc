// Host fast paths for SSE floating point, inline so lifted code gets them
// without a call (sse.h's Arith, Sqrt, MinMax and CvtFloat try these first
// and fall back to the exact SoftFloat paths in sse.cpp).
//
// Under MXCSR's default control (round to nearest, all exceptions masked,
// no FTZ or DAZ), with operands that are normal or zero, an operation whose
// host result is finite and normal gives the same bits on any IEEE-754
// binary32/64 unit rounding to nearest: both are the correctly rounded
// exact value. The only flag such an operation can raise is PE (inexact),
// computed exactly from the rounding error:
// - add/sub: TwoSum, exact for any finite sum;
// - mul/div/sqrt: an FMA residual, exactly representable while the values
//   stay 2^p above the underflow threshold (checked);
// - double to float: whether the float converts back to the same double.
// PE is sticky: once MXCSR has it, inexactness is not computed at all (it
// cannot change the flags). The near-underflow checks stay either way:
// they also keep UE (tininess after rounding) on the exact path.
// A zero result is taken only from add/sub (exact cancellation, its sign
// as IEEE rounding to nearest gives it) or from zero operands. Everything
// else (NaNs, infinities, a zero divisor, denormal operands, negative
// square roots, overflow, results near or below the underflow threshold)
// takes the exact path. MINPS/MAXPS need no arithmetic: with normal or zero
// operands they only compare (equal values, zeros of either sign too, give
// the source). Needs the host's default floating-point environment (round
// to nearest, no flush to zero) and no FMA contraction: lifted code and
// sse.cpp are built with RECOMP_X87_FP_FLAGS. tests/x86/sse_fast_test
// compares every path with the exact one.

#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <type_traits>

#include "x86/ctx.h"

namespace recomp::sse {

// SetFastPath (tests): false sends everything to the exact paths.
extern bool g_fast_path;

namespace fast {

inline constexpr uint32_t kControlMask = 0xFFC0, kDefaultControl = 0x1F80, kPE = 0x20;

template <bool D>
struct Bits {
  using U = std::conditional_t<D, uint64_t, uint32_t>;
  using T = std::conditional_t<D, double, float>;
  static constexpr U kExp = D ? U(0x7FF0000000000000ull) : U(0x7F800000u);
  static constexpr U kFrac = D ? U(0x000FFFFFFFFFFFFFull) : U(0x007FFFFFu);
  // 2^p above the smallest normal: below it an FMA residual could round.
  static constexpr T kSafe = D ? T(0x1p-969) : T(0x1p-102f);
  // For square roots the residual scales with the operand: 2^2p above.
  static constexpr T kSafeSqrt = D ? T(0x1p-916) : T(0x1p-78f);
};

template <bool D>
[[gnu::always_inline]] inline bool NormalOrZero(typename Bits<D>::U v) {
  const typename Bits<D>::U e = v & Bits<D>::kExp;
  return e != Bits<D>::kExp && (e != 0 || !(v & Bits<D>::kFrac));
}

// A finite, normal or zero result.
template <bool D>
[[gnu::always_inline]] inline bool NormalResult(typename Bits<D>::U r) {
  const typename Bits<D>::U e = r & Bits<D>::kExp;
  return e != Bits<D>::kExp && (e != 0 || !(r & Bits<D>::kFrac));
}

template <bool D, int Op>  // Op: sse::Op (kAdd, kSub, kMul, kDiv)
[[gnu::always_inline]] inline bool ArithLane(typename Bits<D>::U a, typename Bits<D>::U b, typename Bits<D>::U* out,
                      bool need_pe, bool* inexact) {
  using T = typename Bits<D>::T;
  using U = typename Bits<D>::U;
  if (!NormalOrZero<D>(a) || !NormalOrZero<D>(b)) return false;
  const T x = std::bit_cast<T>(a), y = std::bit_cast<T>(b);
  T r;
  bool inex = false;
  if constexpr (Op == 0 || Op == 1) {
    const T yy = Op == 0 ? y : -y;
    r = x + yy;
    if (need_pe) {
      const T bv = r - x, av = r - bv;
      inex = ((x - av) + (yy - bv)) != 0;
    }
  } else if constexpr (Op == 2) {
    r = x * y;
    if (r != 0) {
      if (std::fabs(r) < Bits<D>::kSafe) return false;
      if (need_pe) inex = std::fma(x, y, -r) != 0;
    } else if (x != 0 && y != 0) {
      return false;  // underflow to zero
    }
  } else {
    if (y == 0) return false;  // ZE / IE
    r = x / y;
    if (r != 0) {
      if (std::fabs(r) < Bits<D>::kSafe || std::fabs(x) < Bits<D>::kSafe) return false;
      if (need_pe) inex = std::fma(r, y, -x) != 0;
    } else if (x != 0) {
      return false;  // underflow to zero
    }
  }
  const U rb = std::bit_cast<U>(r);
  if (!NormalResult<D>(rb)) return false;  // overflow or underflow
  *out = rb;
  *inexact |= inex;
  return true;
}

template <bool D, int Op>
[[gnu::always_inline]] inline bool Arith(Ctx& c, int lanes, Xmm& d, const Xmm& s) {
  using U = typename Bits<D>::U;
  if (!g_fast_path || (c.mxcsr & kControlMask) != kDefaultControl) return false;
  U r[4];
  bool inexact = false;
  const bool need_pe = !(c.mxcsr & kPE);
  for (int i = 0; i < lanes; i++) {
    U a, b;
    __builtin_memcpy(&a, d.b + i * sizeof(U), sizeof(U));
    __builtin_memcpy(&b, s.b + i * sizeof(U), sizeof(U));
    if (!ArithLane<D, Op>(a, b, &r[i], need_pe, &inexact)) return false;
  }
  __builtin_memcpy(d.b, r, size_t(lanes) * sizeof(U));
  if (inexact) c.mxcsr |= kPE;
  return true;
}

template <bool D>
[[gnu::always_inline]] inline bool Sqrt(Ctx& c, int lanes, Xmm& d, const Xmm& s) {
  using T = typename Bits<D>::T;
  using U = typename Bits<D>::U;
  if (!g_fast_path || (c.mxcsr & kControlMask) != kDefaultControl) return false;
  U r[4];
  bool inexact = false;
  for (int i = 0; i < lanes; i++) {
    U a;
    __builtin_memcpy(&a, s.b + i * sizeof(U), sizeof(U));
    if (!NormalOrZero<D>(a)) return false;
    const T x = std::bit_cast<T>(a);
    if (x < 0) return false;  // IE
    if (x != 0 && x < Bits<D>::kSafeSqrt) return false;
    const T q = std::sqrt(x);  // ±0 stays ±0
    if (x != 0 && !(c.mxcsr & kPE)) inexact |= std::fma(q, q, -x) != 0;
    r[i] = std::bit_cast<U>(q);
  }
  __builtin_memcpy(d.b, r, size_t(lanes) * sizeof(U));
  if (inexact) c.mxcsr |= kPE;
  return true;
}

template <bool D>
[[gnu::always_inline]] inline bool MinMax(Ctx& c, bool max, int lanes, Xmm& d, const Xmm& s) {
  using T = typename Bits<D>::T;
  using U = typename Bits<D>::U;
  if (!g_fast_path || (c.mxcsr & kControlMask) != kDefaultControl) return false;
  U r[4];
  for (int i = 0; i < lanes; i++) {
    U a, b;
    __builtin_memcpy(&a, d.b + i * sizeof(U), sizeof(U));
    __builtin_memcpy(&b, s.b + i * sizeof(U), sizeof(U));
    if (!NormalOrZero<D>(a) || !NormalOrZero<D>(b)) return false;
    const T x = std::bit_cast<T>(a), y = std::bit_cast<T>(b);
    r[i] = (max ? x > y : x < y) ? a : b;
  }
  __builtin_memcpy(d.b, r, size_t(lanes) * sizeof(U));
  return true;
}

[[gnu::always_inline]] inline bool CvtFloat(Ctx& c, bool to_dbl, int lanes, Xmm& d, const Xmm& s) {
  if (!g_fast_path || (c.mxcsr & kControlMask) != kDefaultControl) return false;
  if (to_dbl) {
    uint64_t r[2];
    for (int i = 0; i < lanes; i++) {
      uint32_t a;
      __builtin_memcpy(&a, s.b + i * 4, 4);
      if (!NormalOrZero<false>(a)) return false;
      r[i] = std::bit_cast<uint64_t>(double(std::bit_cast<float>(a)));
    }
    __builtin_memcpy(d.b, r, size_t(lanes) * 8);
    return true;
  }
  uint32_t r[2];
  bool inexact = false;
  for (int i = 0; i < lanes; i++) {
    uint64_t a;
    __builtin_memcpy(&a, s.b + i * 8, 8);
    if (!NormalOrZero<true>(a)) return false;
    const double x = std::bit_cast<double>(a);
    const float f = float(x);
    const uint32_t fb = std::bit_cast<uint32_t>(f);
    if ((fb & 0x7F800000u) == 0x7F800000u) return false;  // overflow
    if (!(fb & 0x7F800000u) && x != 0) return false;      // underflow
    inexact |= double(f) != x;
    r[i] = fb;
  }
  __builtin_memcpy(d.b, r, size_t(lanes) * 4);
  if (lanes == 2) __builtin_memset(d.b + 8, 0, 8);  // CVTPD2PS zeroes the top
  if (inexact) c.mxcsr |= kPE;
  return true;
}

// RSQRTPS/RCPPS lanes (sse.h: the correctly rounded value, no flags).
// For a positive normal x, 1/sqrt(x) computed in binary64 (sqrt and the
// division each correctly rounded) is within 2^-51 relative of the exact
// value, so rounding it to binary32 gives the correctly rounded result
// unless the double lies within a few ulps of a float rounding midpoint:
// those lanes return false for the exact check. (tests/x86/sse_fast_test
// covers every significand at both exponent parities, which by the exact
// scaling rsqrt(4x) = rsqrt(x) / 2 is every normal input: none comes that
// close, so the check is a margin that never fires.)
// 1/x for a normal float is one correctly rounded host division; a result
// below the normal range (flushed by RCPPS) returns false.
[[gnu::always_inline]] inline bool RsqrtLane(uint32_t x, uint32_t* out) {
  const uint32_t e = x & 0x7F800000u;
  if ((x & 0x80000000u) || e == 0 || e == 0x7F800000u) return false;
  const double r = 1.0 / std::sqrt(double(std::bit_cast<float>(x)));
  const uint64_t rb = std::bit_cast<uint64_t>(r);
  const uint64_t low = rb & ((uint64_t(1) << 29) - 1);  // bits float drops
  const uint64_t mid = uint64_t(1) << 28;
  if ((low > mid ? low - mid : mid - low) <= 8) return false;
  *out = std::bit_cast<uint32_t>(float(r));
  return true;
}

[[gnu::always_inline]] inline bool RcpLane(uint32_t x, uint32_t* out) {
  const uint32_t e = x & 0x7F800000u;
  if (e == 0 || e == 0x7F800000u) return false;
  const uint32_t r = std::bit_cast<uint32_t>(1.0f / std::bit_cast<float>(x));
  if (!(r & 0x7F800000u)) return false;  // denormal or zero: flushed
  *out = r;
  return true;
}

}  // namespace fast
}  // namespace recomp::sse

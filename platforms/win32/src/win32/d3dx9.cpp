// d3dx9_35: the D3DX 9 functions games import, on the d3d9 HLE.
//
// Textures from files in memory: DDS (2D and cube; DXT1-5, the common RGB,
// luminance and alpha layouts, and D3DFORMAT FourCCs), PNG (through SDL)
// and JPEG (through ffmpeg, when built with it), both as A8R8G8B8. Missing mip levels
// are generated with a box filter for uncompressed formats. Anything
// else (other image formats, resizing, format conversion) is fatal and
// named, to be added when a game needs it.
//
// Shaders: D3DXCompileShader translates HLSL to GLSL (runtime/hlsl) and
// returns it in a shader function Create*Shader recognise
// (d3d9::TranslatedFunction), with an ID3DXConstantTable of the registers
// the translation placed the uniforms in.

#include <SDL3/SDL.h>

#if RECOMP_D3DX_JPEG
extern "C" {
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
}
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "runtime/log.h"
#include "win32/d3d9_impl.h"
#include "win32/hle.h"
#include "win32/kernel32_state.h"

namespace recomp::hle {
namespace {

using namespace d3d9;
using namespace guest;

constexpr uint32_t D3DX_DEFAULT = 0xFFFFFFFF;
constexpr uint32_t D3DX_DEFAULT_NONPOW2 = 0xFFFFFFFE;
constexpr uint32_t D3DX_FROM_FILE = 0xFFFFFFFD;
constexpr uint32_t D3DFMT_FROM_FILE = 0xFFFFFFFD;
constexpr uint32_t D3DXERR_INVALIDDATA = 0x88760B59;
constexpr uint32_t D3DXIFF_JPG = 1, D3DXIFF_PNG = 3, D3DXIFF_DDS = 4;

struct D3DXIMAGE_INFO {
  uint32_t Width, Height, Depth, MipLevels, Format, ResourceType, ImageFileFormat;
};
static_assert(sizeof(D3DXIMAGE_INFO) == 28);

// DDS_HEADER after the "DDS " magic.
struct DdsHeader {
  uint32_t size, flags, height, width, pitch_or_linear_size, depth, mip_count;
  uint32_t reserved1[11];
  struct {
    uint32_t size, flags, fourcc, rgb_bits, r_mask, g_mask, b_mask, a_mask;
  } pf;
  uint32_t caps, caps2, caps3, caps4, reserved2;
};
static_assert(sizeof(DdsHeader) == 124);

constexpr uint32_t DDPF_ALPHAPIXELS = 0x1, DDPF_ALPHA = 0x2, DDPF_FOURCC = 0x4,
                   DDPF_RGB = 0x40, DDPF_LUMINANCE = 0x20000;
constexpr uint32_t DDSCAPS2_CUBEMAP = 0x200, DDSCAPS2_CUBEMAP_ALLFACES = 0xFC00,
                   DDSCAPS2_VOLUME = 0x200000;

struct Image {
  uint32_t width = 0, height = 0, levels = 0, format = 0;
  bool cube = false;
  const uint8_t* data = nullptr;  // faces, each with its mip chain
  uint32_t size = 0;
};

// The D3DFORMAT of a DDS pixel format, or 0.
uint32_t DdsFormat(const DdsHeader& h) {
  const auto& p = h.pf;
  if (p.flags & DDPF_FOURCC) return p.fourcc;  // DXTn or a D3DFORMAT number
  auto masks = [&](uint32_t bits, uint32_t r, uint32_t g, uint32_t b, uint32_t a) {
    return p.rgb_bits == bits && p.r_mask == r && p.g_mask == g && p.b_mask == b &&
           ((p.flags & DDPF_ALPHAPIXELS) ? p.a_mask : 0) == a;
  };
  if (p.flags & DDPF_RGB) {
    if (masks(32, 0xFF0000, 0xFF00, 0xFF, 0xFF000000)) return D3DFMT_A8R8G8B8;
    if (masks(32, 0xFF0000, 0xFF00, 0xFF, 0)) return D3DFMT_X8R8G8B8;
    if (masks(24, 0xFF0000, 0xFF00, 0xFF, 0)) return D3DFMT_R8G8B8;
    if (masks(16, 0xF800, 0x7E0, 0x1F, 0)) return D3DFMT_R5G6B5;
    if (masks(16, 0x7C00, 0x3E0, 0x1F, 0x8000)) return D3DFMT_A1R5G5B5;
    if (masks(16, 0x7C00, 0x3E0, 0x1F, 0)) return D3DFMT_X1R5G5B5;
    if (masks(16, 0xF00, 0xF0, 0xF, 0xF000)) return D3DFMT_A4R4G4B4;
    if (masks(16, 0xF00, 0xF0, 0xF, 0)) return D3DFMT_X4R4G4B4;
    return 0;
  }
  if (p.flags & DDPF_LUMINANCE) {
    if (p.rgb_bits == 8 && p.r_mask == 0xFF) return D3DFMT_L8;
    if (p.rgb_bits == 16 && p.r_mask == 0xFF && p.a_mask == 0xFF00) return D3DFMT_A8L8;
    return 0;
  }
  if ((p.flags & DDPF_ALPHA) && p.rgb_bits == 8 && p.a_mask == 0xFF) return D3DFMT_A8;
  return 0;
}

uint32_t LevelSize(gfx::Format f, uint32_t w, uint32_t h) {
  return gfx::RowPitch(f, w) * gfx::RowCount(f, h);
}

// Parses a DDS file; false (with *why) if it is not one we read.
bool ParseDds(const uint8_t* p, uint32_t size, Image* img, std::string* why) {
  if (size < 128 || std::memcmp(p, "DDS ", 4) != 0) {
    *why = "not a DDS file";
    return false;
  }
  DdsHeader h;
  std::memcpy(&h, p + 4, sizeof h);
  if (h.size != 124) {
    *why = "bad DDS header size";
    return false;
  }
  if (h.caps2 & DDSCAPS2_VOLUME) {
    *why = "volume DDS";
    return false;
  }
  img->width = h.width;
  img->height = h.height;
  img->levels = std::max(1u, h.mip_count);
  img->format = DdsFormat(h);
  img->cube = (h.caps2 & DDSCAPS2_CUBEMAP) != 0;
  if (img->cube && (h.caps2 & DDSCAPS2_CUBEMAP_ALLFACES) != DDSCAPS2_CUBEMAP_ALLFACES) {
    *why = "cube DDS without all six faces";
    return false;
  }
  gfx::Format f = FormatFromD3D(img->format);
  if (f == gfx::Format::Unknown || f == gfx::Format::P8) {
    char buf[160];
    std::snprintf(buf, sizeof buf,
                  "DDS pixel format flags %X fourcc %08X bits %u masks %X/%X/%X/%X",
                  h.pf.flags, h.pf.fourcc, h.pf.rgb_bits, h.pf.r_mask, h.pf.g_mask,
                  h.pf.b_mask, h.pf.a_mask);
    *why = buf;
    return false;
  }
  uint32_t need = 0;
  for (uint32_t i = 0; i < img->levels; i++)
    need += LevelSize(f, std::max(1u, img->width >> i), std::max(1u, img->height >> i));
  if (img->cube) need *= 6;
  if (size - 128 < need) {
    *why = "truncated DDS";
    return false;
  }
  img->data = p + 128;
  img->size = need;
  return true;
}

// Decodes a PNG into A8R8G8B8 pixels (stored in *pixels); false if it is
// not one SDL reads.
bool ParsePng(const uint8_t* p, uint32_t size, Image* img, std::vector<uint8_t>* pixels) {
  static const uint8_t kSig[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
  if (size < 8 || std::memcmp(p, kSig, 8) != 0) return false;
  SDL_Surface* s = SDL_LoadPNG_IO(SDL_IOFromConstMem(p, size), true);
  if (!s) return false;
  // ARGB8888 is B, G, R, A in memory: D3DFMT_A8R8G8B8.
  SDL_Surface* argb = SDL_ConvertSurface(s, SDL_PIXELFORMAT_ARGB8888);
  SDL_DestroySurface(s);
  if (!argb) return false;
  img->width = uint32_t(argb->w);
  img->height = uint32_t(argb->h);
  img->levels = 1;
  img->format = D3DFMT_A8R8G8B8;
  img->cube = false;
  pixels->resize(size_t(img->width) * img->height * 4);
  for (uint32_t y = 0; y < img->height; y++)
    std::memcpy(pixels->data() + size_t(y) * img->width * 4,
                static_cast<const uint8_t*>(argb->pixels) + size_t(y) * argb->pitch,
                size_t(img->width) * 4);
  SDL_DestroySurface(argb);
  img->data = pixels->data();
  img->size = uint32_t(pixels->size());
  return true;
}

// Decodes a JPEG into A8R8G8B8 pixels (stored in *pixels); false if it is
// not one, or there is no decoder in this build.
bool ParseJpeg(const uint8_t* p, uint32_t size, Image* img, std::vector<uint8_t>* pixels) {
  if (size < 4 || p[0] != 0xFF || p[1] != 0xD8 || p[2] != 0xFF) return false;
#if RECOMP_D3DX_JPEG
  const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_MJPEG);
  AVCodecContext* ctx = codec ? avcodec_alloc_context3(codec) : nullptr;
  AVPacket* pkt = av_packet_alloc();
  AVFrame* frame = av_frame_alloc();
  bool ok = false;
  if (ctx && pkt && frame && avcodec_open2(ctx, codec, nullptr) == 0 &&
      av_new_packet(pkt, int(size)) == 0) {
    std::memcpy(pkt->data, p, size);
    if (avcodec_send_packet(ctx, pkt) == 0 && avcodec_send_packet(ctx, nullptr) >= 0 &&
        avcodec_receive_frame(ctx, frame) == 0) {
      SwsContext* sws = sws_getContext(frame->width, frame->height,
                                       AVPixelFormat(frame->format), frame->width,
                                       frame->height, AV_PIX_FMT_BGRA, SWS_POINT, nullptr,
                                       nullptr, nullptr);
      if (sws) {
        img->width = uint32_t(frame->width);
        img->height = uint32_t(frame->height);
        img->levels = 1;
        img->format = D3DFMT_A8R8G8B8;
        img->cube = false;
        pixels->resize(size_t(img->width) * img->height * 4);
        uint8_t* dst[4] = {pixels->data()};
        int stride[4] = {int(img->width) * 4};
        sws_scale(sws, frame->data, frame->linesize, 0, frame->height, dst, stride);
        sws_freeContext(sws);
        img->data = pixels->data();
        img->size = uint32_t(pixels->size());
        ok = true;
      }
    }
  }
  av_frame_free(&frame);
  av_packet_free(&pkt);
  avcodec_free_context(&ctx);
  return ok;
#else
  (void)img;
  (void)pixels;
  return false;
#endif
}

// The next mip level of an uncompressed image: each pixel the average of
// the 2x2 (or 2x1 / 1x2 at the edges of odd sizes) block above it, per
// channel, through A8R8G8B8.
std::vector<uint8_t> Downsample(const uint8_t* src, uint32_t w, uint32_t h, gfx::Format f) {
  uint32_t nw = std::max(1u, w / 2), nh = std::max(1u, h / 2);
  uint32_t bpp = gfx::BitsPerPixel(f) / 8, pitch = gfx::RowPitch(f, w);
  std::vector<uint8_t> out(size_t(gfx::RowPitch(f, nw)) * nh);
  for (uint32_t y = 0; y < nh; y++)
    for (uint32_t x = 0; x < nw; x++) {
      uint32_t sum[4] = {}, n = 0;
      for (uint32_t dy = 0; dy < 2; dy++)
        for (uint32_t dx = 0; dx < 2; dx++) {
          uint32_t sx = std::min(w - 1, 2 * x + dx), sy = std::min(h - 1, 2 * y + dy);
          uint32_t argb = gfx::DecodePixel(src + size_t(sy) * pitch + size_t(sx) * bpp, f);
          for (int ch = 0; ch < 4; ch++) sum[ch] += (argb >> (8 * ch)) & 0xFF;
          n++;
        }
      uint32_t argb = 0;
      for (int ch = 0; ch < 4; ch++) argb |= ((sum[ch] + n / 2) / n) << (8 * ch);
      gfx::EncodePixel(out.data() + (size_t(y) * nw + x) * bpp, f, argb);
    }
  return out;
}

// DXT1 to DXT2/DXT3, block by block (8 bytes to 16): explicit alpha 0xF
// or, for DXT1's transparent texels, 0 (DXT2: premultiplied, so black).
// DXT2-5 colour blocks always decode in four-colour mode, so a DXT1 block
// in three-colour mode (c0 <= c1) is re-indexed to the nearest of the
// four-colour palette of the same endpoints.
void Dxt1ToDxt3Block(const uint8_t* in, uint8_t* out) {
  uint16_t c0 = uint16_t(in[0] | in[1] << 8), c1 = uint16_t(in[2] | in[3] << 8);
  uint32_t idx = uint32_t(in[4]) | uint32_t(in[5]) << 8 | uint32_t(in[6]) << 16 |
                 uint32_t(in[7]) << 24;
  uint64_t alpha = 0;
  uint32_t nidx = idx;
  if (c0 > c1) {
    alpha = ~0ull;
  } else {
    auto rgb = [](uint16_t c, int ch) {
      int v = ch == 0 ? (c >> 11) & 31 : ch == 1 ? (c >> 5) & 63 : c & 31;
      return ch == 1 ? (v << 2 | v >> 4) : (v << 3 | v >> 2);
    };
    int pal3[3][3], pal4[4][3];
    for (int ch = 0; ch < 3; ch++) {
      int a = rgb(c0, ch), b = rgb(c1, ch);
      pal3[0][ch] = a;
      pal3[1][ch] = b;
      pal3[2][ch] = (a + b) / 2;
      pal4[0][ch] = a;
      pal4[1][ch] = b;
      pal4[2][ch] = (2 * a + b) / 3;
      pal4[3][ch] = (a + 2 * b) / 3;
    }
    nidx = 0;
    for (int i = 0; i < 16; i++) {
      uint32_t k = (idx >> (2 * i)) & 3;
      uint32_t best = 0;
      if (k == 3) {
        // Transparent: alpha 0; premultiplied colour is black, but the
        // texel is invisible either way, so keep index 0.
      } else {
        alpha |= 0xFull << (4 * i);
        int bestd = 1 << 30;
        for (uint32_t j = 0; j < 4; j++) {
          int d = 0;
          for (int ch = 0; ch < 3; ch++) {
            int e = pal3[k][ch] - pal4[j][ch];
            d += e * e;
          }
          if (d < bestd) {
            bestd = d;
            best = j;
          }
        }
      }
      nidx |= best << (2 * i);
    }
  }
  for (int i = 0; i < 8; i++) out[i] = uint8_t(alpha >> (8 * i));
  std::memcpy(out + 8, in, 4);
  for (int i = 0; i < 4; i++) out[12 + i] = uint8_t(nidx >> (8 * i));
}

// ---- DXT compression (uncompressed -> DXT1/3/5) -----------------------------
// A plain bounding-box encoder: colour endpoints are the per-channel min
// and max of the block in RGB565, each texel takes the nearest of the four
// palette colours; DXT5 alpha the same with the eight-step palette.
// Deterministic and good enough for UI art; not D3DX's own compressor.

uint16_t To565(int r, int g, int b) {
  return uint16_t((r * 31 + 127) / 255 << 11 | (g * 63 + 127) / 255 << 5 | (b * 31 + 127) / 255);
}
void From565(uint16_t c, int* rgb) {
  int r = (c >> 11) & 31, g = (c >> 5) & 63, b = c & 31;
  rgb[0] = r << 3 | r >> 2;
  rgb[1] = g << 2 | g >> 4;
  rgb[2] = b << 3 | b >> 2;
}

// 16 ARGB texels -> an 8-byte colour block in four-colour mode.
void EncodeColorBlock(const uint32_t* px, uint8_t* out) {
  int lo[3] = {255, 255, 255}, hi[3] = {0, 0, 0};
  for (int i = 0; i < 16; i++)
    for (int ch = 0; ch < 3; ch++) {
      int v = (px[i] >> (16 - 8 * ch)) & 0xFF;
      lo[ch] = std::min(lo[ch], v);
      hi[ch] = std::max(hi[ch], v);
    }
  uint16_t c0 = To565(hi[0], hi[1], hi[2]), c1 = To565(lo[0], lo[1], lo[2]);
  uint32_t idx = 0;
  if (c0 < c1) std::swap(c0, c1);
  if (c0 != c1) {
    int a[3], b[3], pal[4][3];
    From565(c0, a);
    From565(c1, b);
    for (int ch = 0; ch < 3; ch++) {
      pal[0][ch] = a[ch];
      pal[1][ch] = b[ch];
      pal[2][ch] = (2 * a[ch] + b[ch]) / 3;
      pal[3][ch] = (a[ch] + 2 * b[ch]) / 3;
    }
    for (int i = 0; i < 16; i++) {
      int best = 0, bestd = 1 << 30;
      for (int j = 0; j < 4; j++) {
        int d = 0;
        for (int ch = 0; ch < 3; ch++) {
          int e = int((px[i] >> (16 - 8 * ch)) & 0xFF) - pal[j][ch];
          d += e * e;
        }
        if (d < bestd) {
          bestd = d;
          best = j;
        }
      }
      idx |= uint32_t(best) << (2 * i);
    }
  }
  out[0] = uint8_t(c0);
  out[1] = uint8_t(c0 >> 8);
  out[2] = uint8_t(c1);
  out[3] = uint8_t(c1 >> 8);
  for (int i = 0; i < 4; i++) out[4 + i] = uint8_t(idx >> (8 * i));
}

// 16 ARGB texels -> an 8-byte DXT5 alpha block (eight-alpha mode).
void EncodeAlphaBlock(const uint32_t* px, uint8_t* out) {
  int lo = 255, hi = 0;
  for (int i = 0; i < 16; i++) {
    int a = int(px[i] >> 24);
    lo = std::min(lo, a);
    hi = std::max(hi, a);
  }
  int pal[8];
  pal[0] = hi;
  pal[1] = lo;
  for (int k = 1; k < 7; k++) pal[k + 1] = ((7 - k) * hi + k * lo) / 7;
  uint64_t bits = 0;
  for (int i = 0; i < 16; i++) {
    int a = int(px[i] >> 24), best = 0, bestd = 1 << 30;
    for (int j = 0; j < 8; j++)
      if (std::abs(a - pal[j]) < bestd) {
        bestd = std::abs(a - pal[j]);
        best = j;
      }
    bits |= uint64_t(best) << (3 * i);
  }
  out[0] = uint8_t(hi);
  out[1] = uint8_t(lo);
  for (int i = 0; i < 6; i++) out[2 + i] = uint8_t(bits >> (8 * i));
}

// A DXT1-5 block -> 16 ARGB texels (DXT2/4 stay premultiplied).
void DecodeDxtBlock(const uint8_t* blk, uint32_t fmt, uint32_t* px) {
  const uint8_t* cb = fmt == D3DFMT_DXT1 ? blk : blk + 8;
  uint16_t c0 = uint16_t(cb[0] | cb[1] << 8), c1 = uint16_t(cb[2] | cb[3] << 8);
  int a[3], b[3];
  From565(c0, a);
  From565(c1, b);
  uint32_t pal[4];
  bool four = fmt != D3DFMT_DXT1 || c0 > c1;
  auto pack = [](int r, int g, int bb, uint32_t al) {
    return al << 24 | uint32_t(r) << 16 | uint32_t(g) << 8 | uint32_t(bb);
  };
  pal[0] = pack(a[0], a[1], a[2], 255);
  pal[1] = pack(b[0], b[1], b[2], 255);
  if (four) {
    pal[2] = pack((2 * a[0] + b[0]) / 3, (2 * a[1] + b[1]) / 3, (2 * a[2] + b[2]) / 3, 255);
    pal[3] = pack((a[0] + 2 * b[0]) / 3, (a[1] + 2 * b[1]) / 3, (a[2] + 2 * b[2]) / 3, 255);
  } else {
    pal[2] = pack((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2, 255);
    pal[3] = 0;  // transparent black
  }
  uint32_t idx = uint32_t(cb[4]) | uint32_t(cb[5]) << 8 | uint32_t(cb[6]) << 16 |
                 uint32_t(cb[7]) << 24;
  for (int i = 0; i < 16; i++) px[i] = pal[(idx >> (2 * i)) & 3];
  if (fmt == D3DFMT_DXT2 || fmt == D3DFMT_DXT3) {
    for (int i = 0; i < 16; i++) {
      uint32_t al = (blk[i / 2] >> (4 * (i & 1))) & 0xF;
      px[i] = (px[i] & 0xFFFFFF) | (al * 0x11) << 24;
    }
  } else if (fmt == D3DFMT_DXT4 || fmt == D3DFMT_DXT5) {
    int a0 = blk[0], a1 = blk[1], ap[8];
    ap[0] = a0;
    ap[1] = a1;
    if (a0 > a1) {
      for (int k = 1; k < 7; k++) ap[k + 1] = ((7 - k) * a0 + k * a1) / 7;
    } else {
      for (int k = 1; k < 5; k++) ap[k + 1] = ((5 - k) * a0 + k * a1) / 5;
      ap[6] = 0;
      ap[7] = 255;
    }
    uint64_t bits = 0;
    for (int i = 0; i < 6; i++) bits |= uint64_t(blk[2 + i]) << (8 * i);
    for (int i = 0; i < 16; i++)
      px[i] = (px[i] & 0xFFFFFF) | uint32_t(ap[(bits >> (3 * i)) & 7]) << 24;
  }
}

// A level in format `f` (D3DFORMAT d3d) -> w x h ARGB texels.
std::vector<uint32_t> DecodeLevel(const uint8_t* q, uint32_t w, uint32_t h, uint32_t d3d,
                                  gfx::Format f) {
  std::vector<uint32_t> argb(size_t(w) * h);
  if (gfx::IsCompressed(f)) {
    size_t bs = d3d == D3DFMT_DXT1 ? 8 : 16;
    uint32_t bw = (w + 3) / 4;
    for (uint32_t by = 0; by < (h + 3) / 4; by++)
      for (uint32_t bx = 0; bx < bw; bx++) {
        uint32_t px[16];
        DecodeDxtBlock(q + (size_t(by) * bw + bx) * bs, d3d, px);
        for (uint32_t i = 0; i < 16; i++) {
          uint32_t x = bx * 4 + i % 4, y = by * 4 + i / 4;
          if (x < w && y < h) argb[size_t(y) * w + x] = px[i];
        }
      }
    return argb;
  }
  uint32_t fb = gfx::BitsPerPixel(f) / 8;
  for (uint32_t y = 0; y < h; y++)
    for (uint32_t x = 0; x < w; x++)
      argb[size_t(y) * w + x] =
          gfx::DecodePixel(q + size_t(y) * gfx::RowPitch(f, w) + size_t(x) * fb, f);
  return argb;
}

// A level of ARGB texels (w x h, tightly packed) -> DXT1/3/5 blocks.
void CompressLevel(const uint32_t* argb, uint32_t w, uint32_t h, uint32_t fmt,
                   std::vector<uint8_t>* out) {
  for (uint32_t by = 0; by < h; by += 4)
    for (uint32_t bx = 0; bx < w; bx += 4) {
      uint32_t px[16];
      for (uint32_t i = 0; i < 16; i++) {
        uint32_t x = std::min(w - 1, bx + i % 4), y = std::min(h - 1, by + i / 4);
        px[i] = argb[size_t(y) * w + x];
      }
      uint8_t blk[16];
      size_t n = 8;
      if (fmt == D3DFMT_DXT1) {
        EncodeColorBlock(px, blk);
      } else {
        if (fmt == D3DFMT_DXT5) {
          EncodeAlphaBlock(px, blk);
        } else {  // DXT3: explicit four-bit alpha
          uint64_t a = 0;
          for (int i = 0; i < 16; i++) a |= uint64_t((px[i] >> 28) & 0xF) << (4 * i);
          for (int i = 0; i < 8; i++) blk[i] = uint8_t(a >> (8 * i));
        }
        EncodeColorBlock(px, blk + 8);
        n = 16;
      }
      out->insert(out->end(), blk, blk + n);
    }
}

// The D3DXCreate(Cube)TextureFromFileInMemory(Ex) core.
uint32_t CreateFromMemory(Ctx& c, bool cube, uint32_t device, uint32_t src, uint32_t size,
                          uint32_t width, uint32_t height, uint32_t levels, uint32_t usage,
                          uint32_t format, uint32_t pool, uint32_t color_key,
                          uint32_t info, uint32_t out) {
  Wr32(out, 0);
  if (!EnsureReadable(c, src, size)) return D3DERR_INVALIDCALL;
  const uint8_t* p = reinterpret_cast<const uint8_t*>(GPtr(src));
  Image img;
  std::string why;
  std::vector<uint8_t> png;
  uint32_t file_format = D3DXIFF_DDS;
  if (ParsePng(p, size, &img, &png)) {
    file_format = D3DXIFF_PNG;
  } else if (ParseJpeg(p, size, &img, &png)) {
    file_format = D3DXIFF_JPG;
  } else if (!ParseDds(p, size, &img, &why)) {
    char magic[5] = {};
    std::memcpy(magic, p, std::min<uint32_t>(4, size));
    for (char& ch : magic)
      if (ch && (ch < 32 || ch > 126)) ch = '.';
    FatalF("d3dx9: texture from memory: %s (%u bytes, starts \"%s\" = %02X %02X %02X %02X): "
           "not supported yet",
           why.c_str(), size, magic, size > 0 ? p[0] : 0, size > 1 ? p[1] : 0,
           size > 2 ? p[2] : 0, size > 3 ? p[3] : 0);
  }
  if (img.cube != cube)
    FatalF("d3dx9: a %s DDS for a %s texture", img.cube ? "cube" : "2D",
           cube ? "cube" : "2D");
  if (info) {
    *G<D3DXIMAGE_INFO>(info) = {img.width, img.height, 1, img.levels, img.format,
                                cube ? D3DRTYPE_CUBETEXTURE : D3DRTYPE_TEXTURE, file_format};
  }
  auto from_file = [](uint32_t v) {
    return v == 0 || v == D3DX_DEFAULT || v == D3DX_DEFAULT_NONPOW2 || v == D3DX_FROM_FILE;
  };
  uint32_t w = from_file(width) ? img.width : width;
  uint32_t h = from_file(height) ? img.height : height;
  if (cube) h = w;
  if (w != img.width || h != img.height)
    FatalF("d3dx9: resizing a %ux%u image to %ux%u not supported yet", img.width,
           img.height, w, h);
  uint32_t fmt = (format == D3DFMT_UNKNOWN || format == D3DX_DEFAULT ||
                  format == D3DFMT_FROM_FILE) ? img.format : format;
  // Format conversion (below): DXT1 to DXT2/3 block by block, anything
  // else through A8R8G8B8.
  std::vector<uint8_t> converted;
  if (img.format == D3DFMT_DXT1 && (fmt == D3DFMT_DXT2 || fmt == D3DFMT_DXT3)) {
    gfx::Format from = FormatFromD3D(img.format);
    uint32_t blocks = 0;
    for (uint32_t i = 0; i < img.levels; i++)
      blocks += LevelSize(from, std::max(1u, img.width >> i), std::max(1u, img.height >> i)) / 8;
    blocks *= img.cube ? 6 : 1;
    converted.resize(size_t(blocks) * 16);
    for (uint32_t b = 0; b < blocks; b++)
      Dxt1ToDxt3Block(img.data + size_t(b) * 8, converted.data() + size_t(b) * 16);
    img.data = converted.data();
    img.format = fmt;
  }
  // Any other conversion: each level decoded to A8R8G8B8, then encoded in
  // the target format (DXT1/3/5 by CompressLevel, uncompressed formats
  // EncodePixel writes).
  if (fmt != img.format) {
    gfx::Format from = FormatFromD3D(img.format), to = FormatFromD3D(fmt);
    bool to_dxt = fmt == D3DFMT_DXT1 || fmt == D3DFMT_DXT3 || fmt == D3DFMT_DXT5;
    if (to == gfx::Format::Unknown || (gfx::IsCompressed(to) && !to_dxt) ||
        (!gfx::IsCompressed(to) && gfx::BitsPerPixel(to) % 8 != 0))
      FatalF("d3dx9: converting format %X to %X not supported yet", img.format, fmt);
    std::vector<uint8_t> out;
    const uint8_t* q = img.data;
    for (uint32_t face = 0; face < (img.cube ? 6u : 1u); face++)
      for (uint32_t i = 0; i < img.levels; i++) {
        uint32_t lw = std::max(1u, img.width >> i), lh = std::max(1u, img.height >> i);
        std::vector<uint32_t> argb = DecodeLevel(q, lw, lh, img.format, from);
        if (to_dxt) {
          CompressLevel(argb.data(), lw, lh, fmt, &out);
        } else {
          uint32_t tb = gfx::BitsPerPixel(to) / 8;
          size_t at = out.size();
          out.resize(at + size_t(gfx::RowPitch(to, lw)) * lh);
          for (size_t k = 0; k < argb.size(); k++)
            gfx::EncodePixel(out.data() + at + k * tb, to, argb[k]);
        }
        q += LevelSize(from, lw, lh);
      }
    converted = std::move(out);
    img.data = converted.data();
    img.format = fmt;
  }
  gfx::Format f = FormatFromD3D(fmt);
  uint32_t full = 1;
  for (uint32_t s = std::max(w, h); s > 1; s >>= 1) full++;
  uint32_t want = levels == D3DX_FROM_FILE ? img.levels
                  : (levels == 0 || levels == D3DX_DEFAULT) ? full
                                                            : std::min(levels, full);
  if (want > img.levels && gfx::IsCompressed(f)) {
    // Generating them would mean decompressing and compressing again.
    WarnF("d3dx9: %ux%u DXT texture with %u of %u mip levels: using %u\n", w, h,
          img.levels, want, img.levels);
    want = img.levels;
  }
  if (color_key)
    FatalF("d3dx9: colour key %08X on a texture from memory not supported yet", color_key);

  Device* d = DeviceOf(device);
  uint32_t tex;
  uint32_t hr = NewTexture(*d, device, cube, w, h, want, usage, fmt, pool, &tex);
  if (hr != D3D_OK) return hr;
  Texture* t = com::As<Texture>(tex);
  const uint8_t* q = img.data;
  for (uint32_t face = 0; face < t->Faces(); face++) {
    std::vector<uint8_t> generated;
    for (uint32_t i = 0; i < img.levels; i++) {
      uint32_t lw = std::max(1u, w >> i), lh = std::max(1u, h >> i);
      if (i < want) FillLevel(*t, face, i, q, gfx::RowPitch(f, lw));
      if (i + 1 == img.levels && want > img.levels)
        generated.assign(q, q + LevelSize(f, lw, lh));
      q += LevelSize(f, lw, lh);
    }
    for (uint32_t i = img.levels; i < want; i++) {
      uint32_t pw = std::max(1u, w >> (i - 1)), ph = std::max(1u, h >> (i - 1));
      generated = Downsample(generated.data(), pw, ph, f);
      FillLevel(*t, face, i, generated.data(), gfx::RowPitch(f, std::max(1u, w >> i)));
    }
  }
  Wr32(out, tex);
  RECOMP_TRACE("  texture from memory: DDS %ux%u x%u fmt %X%s -> %u levels\n", img.width,
               img.height, img.levels, img.format, cube ? " cube" : "", want);
  return D3D_OK;
}

// (pDevice, pSrcData, SrcDataSize, ppTexture)
HLE("d3dx9_35.dll", D3DXCreateTextureFromFileInMemory, 4) {
  return CreateFromMemory(c, false, Arg(c, 0), Arg(c, 1), Arg(c, 2), D3DX_DEFAULT,
                          D3DX_DEFAULT, D3DX_DEFAULT, 0, D3DFMT_UNKNOWN, D3DPOOL_MANAGED, 0,
                          0, Arg(c, 3));
}

// (pDevice, pSrcData, SrcDataSize, Width, Height, MipLevels, Usage, Format,
// Pool, Filter, MipFilter, ColorKey, pSrcInfo, pPalette, ppTexture)
HLE("d3dx9_35.dll", D3DXCreateTextureFromFileInMemoryEx, 15) {
  return CreateFromMemory(c, false, Arg(c, 0), Arg(c, 1), Arg(c, 2), Arg(c, 3), Arg(c, 4),
                          Arg(c, 5), Arg(c, 6), Arg(c, 7), Arg(c, 8), Arg(c, 11), Arg(c, 12),
                          Arg(c, 14));
}

// (pDevice, pSrcData, SrcDataSize, Width, Height, Depth, MipLevels, Usage,
// Format, Pool, Filter, MipFilter, ColorKey, pSrcInfo, pPalette,
// ppVolumeTexture): volume DDS files, at their own size and format.
HLE("d3dx9_35.dll", D3DXCreateVolumeTextureFromFileInMemoryEx, 16) {
  uint32_t device = Arg(c, 0), src = Arg(c, 1), size = Arg(c, 2), out = Arg(c, 15);
  Wr32(out, 0);
  if (!EnsureReadable(c, src, size)) return D3DERR_INVALIDCALL;
  const uint8_t* p = reinterpret_cast<const uint8_t*>(GPtr(src));
  DdsHeader h;
  if (size < 128 || std::memcmp(p, "DDS ", 4) != 0) return D3DXERR_INVALIDDATA;
  std::memcpy(&h, p + 4, sizeof h);
  // A volume DDS has DDSCAPS2_VOLUME (or DDSD_DEPTH) and a depth; D3DX
  // loads a 2D DDS as a volume of depth 1.
  constexpr uint32_t DDSD_DEPTH = 0x800000;
  if (h.size != 124) return D3DXERR_INVALIDDATA;
  if (h.caps2 & DDSCAPS2_CUBEMAP) FatalF("d3dx9: a cube DDS for a volume texture");
  bool volume = (h.caps2 & DDSCAPS2_VOLUME) || (h.flags & DDSD_DEPTH);
  uint32_t fmt = DdsFormat(h);
  gfx::Format f = FormatFromD3D(fmt);
  if (f == gfx::Format::Unknown) FatalF("d3dx9: volume DDS pixel format not supported");
  uint32_t w = h.width, ht = h.height, depth = volume ? std::max(1u, h.depth) : 1;
  uint32_t levels = std::max(1u, h.mip_count);
  uint32_t need = 0;
  for (uint32_t i = 0; i < levels; i++)
    need += LevelSize(f, std::max(1u, w >> i), std::max(1u, ht >> i)) *
            std::max(1u, depth >> i);
  if (size - 128 < need) return D3DXERR_INVALIDDATA;
  auto from_file = [](uint32_t v) {
    return v == 0 || v == D3DX_DEFAULT || v == D3DX_DEFAULT_NONPOW2 || v == D3DX_FROM_FILE;
  };
  if ((!from_file(Arg(c, 3)) && Arg(c, 3) != w) || (!from_file(Arg(c, 4)) && Arg(c, 4) != ht) ||
      (!from_file(Arg(c, 5)) && Arg(c, 5) != depth))
    FatalF("d3dx9: resizing a %ux%ux%u volume not supported yet", w, ht, depth);
  uint32_t format = Arg(c, 8);
  if (format != D3DFMT_UNKNOWN && format != D3DX_DEFAULT && format != D3DFMT_FROM_FILE &&
      format != fmt)
    FatalF("d3dx9: converting volume format %X to %X not supported yet", fmt, format);
  if (uint32_t info = Arg(c, 13))
    *G<D3DXIMAGE_INFO>(info) = {w, ht, depth, levels, fmt, D3DRTYPE_VOLUMETEXTURE, D3DXIFF_DDS};
  Device* d = DeviceOf(device);
  uint32_t tex;
  uint32_t hr = NewVolumeTexture(*d, device, w, ht, depth, levels, Arg(c, 7), fmt, Arg(c, 9), &tex);
  if (hr != D3D_OK) return hr;
  VolumeTexture* t = com::As<VolumeTexture>(tex);
  const uint8_t* q = p + 128;
  for (uint32_t i = 0; i < t->levels; i++) {
    uint32_t lw = std::max(1u, w >> i), lh = std::max(1u, ht >> i);
    uint32_t slice = LevelSize(f, lw, lh);
    FillVolumeLevel(*t, i, q, gfx::RowPitch(f, lw), slice);
    q += slice * std::max(1u, depth >> i);
  }
  Wr32(out, tex);
  RECOMP_TRACE("  volume texture from memory: DDS %ux%ux%u x%u fmt %X\n", w, ht, depth, levels,
               fmt);
  return D3D_OK;
}

// (pDevice, pSrcData, SrcDataSize, ppCubeTexture)
HLE("d3dx9_35.dll", D3DXCreateCubeTextureFromFileInMemory, 4) {
  return CreateFromMemory(c, true, Arg(c, 0), Arg(c, 1), Arg(c, 2), D3DX_DEFAULT,
                          D3DX_DEFAULT, D3DX_DEFAULT, 0, D3DFMT_UNKNOWN, D3DPOOL_MANAGED, 0,
                          0, Arg(c, 3));
}

// (pDevice, pSrcData, SrcDataSize, Size, MipLevels, Usage, Format, Pool,
// Filter, MipFilter, ColorKey, pSrcInfo, pPalette, ppCubeTexture)
HLE("d3dx9_35.dll", D3DXCreateCubeTextureFromFileInMemoryEx, 14) {
  return CreateFromMemory(c, true, Arg(c, 0), Arg(c, 1), Arg(c, 2), Arg(c, 3), Arg(c, 3),
                          Arg(c, 4), Arg(c, 5), Arg(c, 6), Arg(c, 7), Arg(c, 10), Arg(c, 11),
                          Arg(c, 13));
}

// ---- maths -----------------------------------------------------------------------------
// In single precision, operations in the order D3DX evaluates them (built
// without FMA contraction, see CMakeLists.txt): the game's own maths uses
// the results. D3DXMATRIX: 16 floats, row-major, row vectors.

using Mat = std::array<float, 16>;

Mat ReadMatrix(uint32_t va) {
  Mat m;
  std::memcpy(m.data(), GPtr(va), sizeof m);
  return m;
}

void WriteMatrix(uint32_t va, const Mat& m) { std::memcpy(GPtr(va), m.data(), sizeof m); }

struct Vec3 {
  float x, y, z;
};

Vec3 ReadVec3(uint32_t va) {
  Vec3 v;
  std::memcpy(&v, GPtr(va), sizeof v);
  return v;
}

Vec3 Sub(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
float Dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 Cross(Vec3 a, Vec3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
// D3DXVec3Normalize: a zero vector stays zero.
Vec3 Normalize(Vec3 v) {
  float len = std::sqrt(Dot(v, v));
  if (len == 0.0f) return {0, 0, 0};
  return {v.x / len, v.y / len, v.z / len};
}

// (pOut, pM1, pM2): pOut may be either input.
HLE("d3dx9_35.dll", D3DXMatrixMultiply, 3) {
  Mat a = ReadMatrix(Arg(c, 1)), b = ReadMatrix(Arg(c, 2)), r;
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      r[i * 4 + j] = a[i * 4] * b[j] + a[i * 4 + 1] * b[4 + j] + a[i * 4 + 2] * b[8 + j] +
                     a[i * 4 + 3] * b[12 + j];
  WriteMatrix(Arg(c, 0), r);
  return Arg(c, 0);
}

// (pOut, pEye, pAt, pUp)
HLE("d3dx9_35.dll", D3DXMatrixLookAtLH, 4) {
  Vec3 eye = ReadVec3(Arg(c, 1)), at = ReadVec3(Arg(c, 2)), up = ReadVec3(Arg(c, 3));
  Vec3 z = Normalize(Sub(at, eye));
  Vec3 x = Normalize(Cross(up, z));
  Vec3 y = Cross(z, x);
  Mat m = {x.x, y.x, z.x, 0, x.y, y.y, z.y, 0, x.z, y.z, z.z, 0,
           -Dot(x, eye), -Dot(y, eye), -Dot(z, eye), 1};
  WriteMatrix(Arg(c, 0), m);
  return Arg(c, 0);
}

// (pOut, w, h, zn, zf)
HLE("d3dx9_35.dll", D3DXMatrixOrthoLH, 5) {
  auto f = [&](int i) {
    uint32_t v = Arg(c, i);
    float x;
    std::memcpy(&x, &v, 4);
    return x;
  };
  float w = f(1), h = f(2), zn = f(3), zf = f(4);
  Mat m{};
  m[0] = 2.0f / w;
  m[5] = 2.0f / h;
  m[10] = 1.0f / (zf - zn);
  m[14] = zn / (zn - zf);
  m[15] = 1.0f;
  WriteMatrix(Arg(c, 0), m);
  return Arg(c, 0);
}

// (pOut, w, h, zn, zf): right-handed; z maps to [0, 1] from -zn to -zf.
HLE("d3dx9_35.dll", D3DXMatrixOrthoRH, 5) {
  auto f = [&](int i) {
    uint32_t v = Arg(c, i);
    float x;
    std::memcpy(&x, &v, 4);
    return x;
  };
  float w = f(1), h = f(2), zn = f(3), zf = f(4);
  Mat m{};
  m[0] = 2.0f / w;
  m[5] = 2.0f / h;
  m[10] = 1.0f / (zn - zf);
  m[14] = zn / (zn - zf);
  m[15] = 1.0f;
  WriteMatrix(Arg(c, 0), m);
  return Arg(c, 0);
}

// (pOut, pEye, pAt, pUp): right-handed, z axis from the target to the eye.
HLE("d3dx9_35.dll", D3DXMatrixLookAtRH, 4) {
  Vec3 eye = ReadVec3(Arg(c, 1)), at = ReadVec3(Arg(c, 2)), up = ReadVec3(Arg(c, 3));
  Vec3 z = Normalize(Sub(eye, at));
  Vec3 x = Normalize(Cross(up, z));
  Vec3 y = Cross(z, x);
  Mat m = {x.x, y.x, z.x, 0, x.y, y.y, z.y, 0, x.z, y.z, z.z, 0,
           -Dot(x, eye), -Dot(y, eye), -Dot(z, eye), 1};
  WriteMatrix(Arg(c, 0), m);
  return Arg(c, 0);
}

// (pOut, pV, pM): (x, y, z, 1) * M, divided by the resulting w.
HLE("d3dx9_35.dll", D3DXVec3TransformCoord, 3) {
  Vec3 v = ReadVec3(Arg(c, 1));
  Mat m = ReadMatrix(Arg(c, 2));
  float x = v.x * m[0] + v.y * m[4] + v.z * m[8] + m[12];
  float y = v.x * m[1] + v.y * m[5] + v.z * m[9] + m[13];
  float z = v.x * m[2] + v.y * m[6] + v.z * m[10] + m[14];
  float w = v.x * m[3] + v.y * m[7] + v.z * m[11] + m[15];
  float r[3] = {x / w, y / w, z / w};
  std::memcpy(GPtr(Arg(c, 0)), r, sizeof r);
  return Arg(c, 0);
}

// ---- ID3DXBuffer ------------------------------------------------------------------------

struct XBuffer : com::Object {
  uint32_t mem = 0, size = 0;
  ~XBuffer() override {
    if (mem) HeapFreeProcess(mem);
  }
};

uint32_t XBufferVtable() {
  return com::Vtable("d3dx9_35.dll", "ID3DXBuffer",
                     {"QueryInterface", "AddRef", "Release", "GetBufferPointer", "GetBufferSize"});
}

// A new ID3DXBuffer of `size` bytes, copied from `data` if given (else zero).
uint32_t NewXBuffer(const void* data, uint32_t size) {
  auto b = std::make_shared<XBuffer>();
  b->size = size;
  b->mem = HeapAllocProcess(std::max(size, 1u), true);
  if (data) std::memcpy(GPtr(b->mem), data, size);
  return com::Publish(b, XBufferVtable());
}

XBuffer* XBuf(Ctx& c) {
  auto* b = com::As<XBuffer>(Arg(c, 0));
  if (!b) FatalF("ID3DXBuffer call on unknown object %08X", uint32_t(Arg(c, 0)));
  return b;
}

HLE_NAMED("d3dx9_35.dll", "ID3DXBuffer::AddRef", D3DXBuffer_AddRef, 1) { return com::AddRef(Arg(c, 0)); }
HLE_NAMED("d3dx9_35.dll", "ID3DXBuffer::Release", D3DXBuffer_Release, 1) { return com::Release(Arg(c, 0)); }
HLE_NAMED("d3dx9_35.dll", "ID3DXBuffer::GetBufferPointer", D3DXBuffer_GetBufferPointer, 1) {
  return XBuf(c)->mem;
}
HLE_NAMED("d3dx9_35.dll", "ID3DXBuffer::GetBufferSize", D3DXBuffer_GetBufferSize, 1) {
  return XBuf(c)->size;
}

// (NumBytes, ppBuffer)
HLE("d3dx9_35.dll", D3DXCreateBuffer, 2) {
  if (!Arg(c, 1)) return D3DERR_INVALIDCALL;
  Wr32(Arg(c, 1), NewXBuffer(nullptr, Arg(c, 0)));
  return D3D_OK;
}

// ---- ID3DXConstantTable -----------------------------------------------------------------
// Top-level constants only (the translator has no struct uniforms). A
// D3DXHANDLE is an address inside `handles`, one byte per constant; like
// D3DX, any other value passed as a handle is read as a constant's name.

struct ConstantTable : com::Object {
  uint32_t version = 0;  // the shader's version token
  std::vector<hlsl::Constant> constants;
  uint32_t handles = 0;
  std::vector<uint32_t> names;  // guest strings
  uint32_t creator = 0;
  ~ConstantTable() override {
    if (handles) HeapFreeProcess(handles);
    for (uint32_t n : names) HeapFreeProcess(n);
    if (creator) HeapFreeProcess(creator);
  }
};

uint32_t ConstantTableVtable() {
  return com::Vtable(
      "d3dx9_35.dll", "ID3DXConstantTable",
      {"QueryInterface", "AddRef", "Release", "GetBufferPointer", "GetBufferSize", "GetDesc",
       "GetConstantDesc", "GetSamplerIndex", "GetConstant", "GetConstantByName",
       "GetConstantElement", "SetDefaults", "SetValue", "SetBool", "SetBoolArray", "SetInt",
       "SetIntArray", "SetFloat", "SetFloatArray", "SetVector", "SetVectorArray", "SetMatrix",
       "SetMatrixArray", "SetMatrixPointerArray", "SetMatrixTranspose",
       "SetMatrixTransposeArray", "SetMatrixTransposePointerArray"});
}

uint32_t GuestString(const std::string& s) {
  uint32_t p = HeapAllocProcess(uint32_t(s.size()) + 1, true);
  std::memcpy(GPtr(p), s.c_str(), s.size() + 1);
  return p;
}

uint32_t NewConstantTable(const hlsl::Program& p, uint32_t version) {
  auto t = std::make_shared<ConstantTable>();
  t->version = version;
  t->constants = p.constants;
  t->handles = HeapAllocProcess(std::max<uint32_t>(uint32_t(p.constants.size()), 1), true);
  for (const auto& k : p.constants) t->names.push_back(GuestString(k.name));
  t->creator = GuestString("SwARc HLSL translator");
  return com::Publish(t, ConstantTableVtable());
}

ConstantTable* Table(Ctx& c) {
  auto* t = com::As<ConstantTable>(Arg(c, 0));
  if (!t) FatalF("ID3DXConstantTable call on unknown object %08X", uint32_t(Arg(c, 0)));
  return t;
}

// The constant a handle (or name) refers to, -1 if none.
int FindConstant(const ConstantTable& t, uint32_t h) {
  if (!h) return -1;
  if (h >= t.handles && h < t.handles + t.constants.size()) return int(h - t.handles);
  std::string name = GStr(h);
  for (size_t i = 0; i < t.constants.size(); i++)
    if (t.constants[i].name == name) return int(i);
  return -1;
}

HLE_NAMED("d3dx9_35.dll", "ID3DXConstantTable::AddRef", D3DXConstTable_AddRef, 1) {
  return com::AddRef(Arg(c, 0));
}
HLE_NAMED("d3dx9_35.dll", "ID3DXConstantTable::Release", D3DXConstTable_Release, 1) {
  return com::Release(Arg(c, 0));
}

// (pDesc): D3DXCONSTANTTABLE_DESC {Creator, Version, Constants}
HLE_NAMED("d3dx9_35.dll", "ID3DXConstantTable::GetDesc", D3DXConstTable_GetDesc, 2) {
  ConstantTable* t = Table(c);
  uint32_t d = Arg(c, 1);
  if (!d) return D3DERR_INVALIDCALL;
  Wr32(d, t->creator);
  Wr32(d + 4, t->version);
  Wr32(d + 8, uint32_t(t->constants.size()));
  return D3D_OK;
}

// (hConstant, pConstantDesc, pCount): D3DXCONSTANT_DESC, 48 bytes each.
HLE_NAMED("d3dx9_35.dll", "ID3DXConstantTable::GetConstantDesc", D3DXConstTable_GetConstantDesc, 4) {
  ConstantTable* t = Table(c);
  int i = FindConstant(*t, Arg(c, 1));
  uint32_t d = Arg(c, 2), count = Arg(c, 3);
  if (i < 0 || !count) return D3DERR_INVALIDCALL;
  if (d && Rd32(count) >= 1) {
    const hlsl::Constant& k = t->constants[i];
    const uint32_t v[12] = {t->names[i], uint32_t(k.set), k.index, k.count, uint32_t(k.cls),
                            uint32_t(k.type), k.rows, k.columns, k.elements, k.struct_members,
                            k.bytes, 0 /* DefaultValue */};
    for (int w = 0; w < 12; w++) Wr32(d + 4 * w, v[w]);
  }
  Wr32(count, 1);
  return D3D_OK;
}

// (hConstant): the sampler register, or -1.
HLE_NAMED("d3dx9_35.dll", "ID3DXConstantTable::GetSamplerIndex", D3DXConstTable_GetSamplerIndex, 2) {
  ConstantTable* t = Table(c);
  int i = FindConstant(*t, Arg(c, 1));
  if (i < 0 || t->constants[i].set != hlsl::RegisterSet::Sampler) return 0xFFFFFFFF;
  return t->constants[i].index;
}

// (hConstant, Index): the Index-th top-level constant when hConstant is
// null.
HLE_NAMED("d3dx9_35.dll", "ID3DXConstantTable::GetConstant", D3DXConstTable_GetConstant, 3) {
  ConstantTable* t = Table(c);
  if (Arg(c, 1)) FatalF("ID3DXConstantTable::GetConstant of a struct member");
  return Arg(c, 2) < t->constants.size() ? t->handles + Arg(c, 2) : 0;
}

// (hConstant, pName)
HLE_NAMED("d3dx9_35.dll", "ID3DXConstantTable::GetConstantByName", D3DXConstTable_GetConstantByName, 3) {
  ConstantTable* t = Table(c);
  if (Arg(c, 1)) FatalF("ID3DXConstantTable::GetConstantByName of a struct member");
  if (!Arg(c, 2)) return 0;
  std::string name = GStr(Arg(c, 2));
  for (size_t i = 0; i < t->constants.size(); i++)
    if (t->constants[i].name == name) return t->handles + uint32_t(i);
  return 0;
}

// ---- shaders --------------------------------------------------------------------------

// <ID>_SHADER_DUMP=dir: every HLSL source D3DXCompileShader gets is
// written to dir/<FNV-1a of profile, entry and source>.hlsl, with the
// profile, entry point and macros in a comment on top (the format
// recomp_hlslc reads).
void DumpShader(const std::string& src, const std::string& profile, const std::string& entry,
                const std::string& defines) {
  const char* dir = Env("SHADER_DUMP");
  if (!dir || !*dir) return;
  uint64_t h = 1469598103934665603ull;
  for (const std::string* s : {&profile, &entry, &src})
    for (unsigned char ch : *s) h = (h ^ ch) * 1099511628211ull;
  char path[512];
  std::snprintf(path, sizeof path, "%s/%016llx.hlsl", dir, (unsigned long long)h);
  if (FILE* f = std::fopen(path, "wb")) {
    std::fprintf(f, "// profile %s, entry %s%s%s\n", profile.c_str(), entry.c_str(),
                 defines.empty() ? "" : ", defines ", defines.c_str());
    std::fwrite(src.data(), 1, src.size(), f);
    std::fclose(f);
  }
}

// (pSrcData, SrcDataLen, pDefines, pInclude, pFunctionName, pProfile,
// Flags, ppShader, ppErrorMsgs, ppConstantTable)
HLE("d3dx9_35.dll", D3DXCompileShader, 10) {
  uint32_t src = Arg(c, 0), len = Arg(c, 1), defs = Arg(c, 2);
  if (!EnsureReadable(c, src, len)) return D3DERR_INVALIDCALL;
  std::string text(reinterpret_cast<const char*>(GPtr(src)), len);
  std::vector<hlsl::Define> macros;
  std::string defines;
  for (uint32_t d = defs; d && Rd32(d); d += 8) {  // D3DXMACRO {Name, Definition}
    hlsl::Define m{GStr(Rd32(d)), Rd32(d + 4) ? GStr(Rd32(d + 4)) : ""};
    if (!defines.empty()) defines += ' ';
    defines += m.name + (m.value.empty() ? "" : "=" + m.value);
    macros.push_back(std::move(m));
  }
  std::string entry = GStr(Arg(c, 4)), profile = GStr(Arg(c, 5));
  uint32_t flags = Arg(c, 6);
  DumpShader(text, profile, entry, defines);
  if (Arg(c, 3)) FatalF("d3dx9: D3DXCompileShader with an include handler not supported yet");
  hlsl::Program prog;
  std::string errors;
  // A source the original compiler accepts and the translator does not is
  // a translator gap: fatal, with the message (dump the source to see it).
  if (!hlsl::Compile(text, macros, entry, profile, flags, &prog, &errors))
    FatalF("d3dx9: D3DXCompileShader(%s, %s, %u bytes, flags %X) failed: %s", profile.c_str(),
           entry.c_str(), len, flags, errors.c_str());
  std::vector<uint32_t> fn = TranslatedFunction(prog);
  if (Arg(c, 7)) Wr32(Arg(c, 7), NewXBuffer(fn.data(), uint32_t(fn.size() * 4)));
  if (Arg(c, 8)) Wr32(Arg(c, 8), 0);  // no messages
  if (Arg(c, 9)) Wr32(Arg(c, 9), NewConstantTable(prog, fn[0]));
  return D3D_OK;
}

}  // namespace
}  // namespace recomp::hle

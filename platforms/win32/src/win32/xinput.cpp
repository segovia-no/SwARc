// XInput 1.3 (xinput1_3.dll, by ordinal) on SDL gamepads. Up to four pads
// take the user indices in the order they connect and keep them until
// they disconnect. SDL's gamepad layout is the Xbox controller's (south =
// A, east = B, west = X, north = Y); stick Y is up in XInput and down in
// SDL. Automated runs (<ID>_BACKGROUND) see no pads, as they take no host
// input. The gamepad subsystem starts on the main thread, where SDL's
// events are pumped (callers on other threads see no pads until then).
// DirectInput and WMI list the same pads (xinput.h). Controllers on an
// Xbox 360 wireless receiver the host has no driver for are read directly
// (xbox360_receiver.h) and take user indices the same way, with their own
// XInput subtype (guitars, drums...).

#include <SDL3/SDL.h>

#include <algorithm>
#include <cstring>
#include <mutex>

#include "runtime/log.h"
#include "win32/hle.h"
#include "win32/user32_window.h"
#include "win32/xbox360_receiver.h"
#include "win32/xinput.h"

namespace recomp::hle {
namespace {

constexpr uint32_t ERROR_SUCCESS = 0, ERROR_BAD_ARGUMENTS = 160,
                   ERROR_DEVICE_NOT_CONNECTED = 1167;

struct Pad {
  SDL_Gamepad* gp = nullptr;
  int rx = -1;  // receiver slot (gp null)
  uint32_t packet = 0;
  uint8_t last[12] = {};  // XINPUT_GAMEPAD last reported
};

std::mutex g_mu;
bool g_up = false;
bool g_enabled = true;  // XInputEnable
Pad g_pads[4];

// Gives receiver controllers that connected free user indices and frees
// those that disconnected; lights each one's quadrant for its index.
void RefreshReceiver() {
  x360rx::Start();
  bool taken[x360rx::kSlots] = {};
  for (uint32_t i = 0; i < 4; i++) {
    Pad& p = g_pads[i];
    if (p.rx < 0) continue;
    if (!x360rx::Read(p.rx, nullptr)) {
      p = Pad{};
      continue;
    }
    taken[p.rx] = true;
    x360rx::SetLed(p.rx, int(i));
  }
  for (int s = 0; s < x360rx::kSlots; s++) {
    if (taken[s] || !x360rx::Read(s, nullptr)) continue;
    for (uint32_t i = 0; i < 4; i++) {
      Pad& p = g_pads[i];
      if (p.gp || p.rx >= 0) continue;
      p.rx = s;
      x360rx::SetLed(s, int(i));
      break;
    }
  }
}

// Opens newly connected gamepads into free slots, frees disconnected ones.
void Refresh() {
  if (Background()) return;
  RefreshReceiver();
  if (!g_up) {
    if (!OnMainThread() || Background()) return;
    if (!SDL_InitSubSystem(SDL_INIT_GAMEPAD)) {
      WarnF("xinput: SDL gamepad init failed: %s\n", SDL_GetError());
      return;
    }
    g_up = true;
  }
  for (Pad& p : g_pads) {
    if (p.gp && !SDL_GamepadConnected(p.gp)) {
      SDL_CloseGamepad(p.gp);
      p = Pad{};
    }
  }
  int n = 0;
  SDL_JoystickID* ids = SDL_GetGamepads(&n);
  for (int i = 0; ids && i < n; i++) {
    bool open = false;
    for (const Pad& p : g_pads)
      if (p.gp && SDL_GetGamepadID(p.gp) == ids[i]) open = true;
    if (open) continue;
    for (Pad& p : g_pads) {
      if (p.gp || p.rx >= 0) continue;
      p.gp = SDL_OpenGamepad(ids[i]);
      break;
    }
  }
  SDL_free(ids);
}

// The pad at `index`, refreshed; null if none.
Pad* PadAt(uint32_t index) {
  if (index >= 4) return nullptr;
  Refresh();
  return g_pads[index].gp || g_pads[index].rx >= 0 ? &g_pads[index] : nullptr;
}

int16_t Stick(SDL_Gamepad* gp, SDL_GamepadAxis a, bool flip) {
  int v = SDL_GetGamepadAxis(gp, a);
  if (flip) v = -v;
  return int16_t(std::clamp(v, -32768, 32767));
}

uint8_t Trigger(SDL_Gamepad* gp, SDL_GamepadAxis a) {
  int v = SDL_GetGamepadAxis(gp, a);
  return uint8_t(std::clamp(v, 0, 32767) * 255 / 32767);
}

// XINPUT_GAMEPAD: wButtons, bLeftTrigger, bRightTrigger, sThumbLX/LY/RX/RY.
void ReadGamepad(SDL_Gamepad* gp, uint8_t out[12]) {
  static const std::pair<SDL_GamepadButton, uint16_t> kButtons[] = {
      {SDL_GAMEPAD_BUTTON_DPAD_UP, 0x0001},     {SDL_GAMEPAD_BUTTON_DPAD_DOWN, 0x0002},
      {SDL_GAMEPAD_BUTTON_DPAD_LEFT, 0x0004},   {SDL_GAMEPAD_BUTTON_DPAD_RIGHT, 0x0008},
      {SDL_GAMEPAD_BUTTON_START, 0x0010},       {SDL_GAMEPAD_BUTTON_BACK, 0x0020},
      {SDL_GAMEPAD_BUTTON_LEFT_STICK, 0x0040},  {SDL_GAMEPAD_BUTTON_RIGHT_STICK, 0x0080},
      {SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, 0x0100}, {SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, 0x0200},
      {SDL_GAMEPAD_BUTTON_SOUTH, 0x1000},       {SDL_GAMEPAD_BUTTON_EAST, 0x2000},
      {SDL_GAMEPAD_BUTTON_WEST, 0x4000},        {SDL_GAMEPAD_BUTTON_NORTH, 0x8000},
  };
  uint16_t buttons = 0;
  for (const auto& [b, bit] : kButtons)
    if (SDL_GetGamepadButton(gp, b)) buttons |= bit;
  int16_t sticks[4] = {Stick(gp, SDL_GAMEPAD_AXIS_LEFTX, false),
                       Stick(gp, SDL_GAMEPAD_AXIS_LEFTY, true),
                       Stick(gp, SDL_GAMEPAD_AXIS_RIGHTX, false),
                       Stick(gp, SDL_GAMEPAD_AXIS_RIGHTY, true)};
  std::memcpy(out, &buttons, 2);
  out[2] = Trigger(gp, SDL_GAMEPAD_AXIS_LEFT_TRIGGER);
  out[3] = Trigger(gp, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER);
  std::memcpy(out + 4, sticks, 8);
}

// XInputGetState(dwUserIndex, pState): XINPUT_STATE {dwPacketNumber,
// XINPUT_GAMEPAD}; the packet number changes when the state does.
HLE_NAMED("xinput1_3.dll", "#2", XInputGetState, 2) {
  std::lock_guard<std::mutex> l(g_mu);
  Pad* p = PadAt(Arg(c, 0));
  if (!p) return ERROR_DEVICE_NOT_CONNECTED;
  if (!Arg(c, 1)) return ERROR_BAD_ARGUMENTS;
  uint8_t g[12] = {};
  if (g_enabled) {
    x360rx::SlotState rs;
    if (p->gp)
      ReadGamepad(p->gp, g);
    else if (x360rx::Read(p->rx, &rs))
      std::memcpy(g, rs.gamepad, sizeof g);
  }
  if (std::memcmp(g, p->last, sizeof g) != 0) {
    std::memcpy(p->last, g, sizeof g);
    p->packet++;
  }
  Wr32(Arg(c, 1), p->packet);
  std::memcpy(GPtr(Arg(c, 1) + 4), g, sizeof g);
  return ERROR_SUCCESS;
}

// XInputEnable(enable): games call it with FALSE when they lose focus.
// While disabled, pads stay connected but report a neutral state.
HLE_NAMED("xinput1_3.dll", "#5", XInputEnable, 1) {
  std::lock_guard<std::mutex> l(g_mu);
  g_enabled = Arg(c, 0) != 0;
  return 0;
}

// XInputSetState(dwUserIndex, pVibration): XINPUT_VIBRATION
// {wLeftMotorSpeed, wRightMotorSpeed}; held until changed.
HLE_NAMED("xinput1_3.dll", "#3", XInputSetState, 2) {
  std::lock_guard<std::mutex> l(g_mu);
  Pad* p = PadAt(Arg(c, 0));
  if (!p) return ERROR_DEVICE_NOT_CONNECTED;
  if (!Arg(c, 1)) return ERROR_BAD_ARGUMENTS;
  uint32_t v = Rd32(Arg(c, 1));
  if (p->gp)
    SDL_RumbleGamepad(p->gp, uint16_t(v), uint16_t(v >> 16), 0xFFFFFFFF);
  else
    x360rx::SetRumble(p->rx, uint8_t(v >> 8), uint8_t(v >> 24));
  return ERROR_SUCCESS;
}

// XInputGetCapabilities(dwUserIndex, dwFlags, pCapabilities):
// XINPUT_CAPABILITIES {Type, SubType, Flags, Gamepad, Vibration}: every
// control and both motors; the subtype is a gamepad's, or for a receiver
// controller the one it announced.
HLE_NAMED("xinput1_3.dll", "#4", XInputGetCapabilities, 3) {
  std::lock_guard<std::mutex> l(g_mu);
  Pad* p = PadAt(Arg(c, 0));
  if (!p) return ERROR_DEVICE_NOT_CONNECTED;
  uint32_t out = Arg(c, 2);
  if (!out || Arg(c, 1) > 1) return ERROR_BAD_ARGUMENTS;
  uint8_t caps[20] = {};
  caps[0] = 1;  // XINPUT_DEVTYPE_GAMEPAD
  caps[1] = 1;  // XINPUT_DEVSUBTYPE_GAMEPAD
  x360rx::SlotState rs;
  if (!p->gp && x360rx::Read(p->rx, &rs)) caps[1] = rs.subtype;
  const uint16_t buttons = 0xF3FF;
  std::memcpy(caps + 4, &buttons, 2);
  caps[6] = caps[7] = 0xFF;
  std::memset(caps + 8, 0xFF, 8);    // stick ranges
  std::memset(caps + 16, 0xFF, 4);   // motors
  std::memcpy(GPtr(out), caps, sizeof caps);
  return ERROR_SUCCESS;
}

}  // namespace

std::vector<XInputPad> XInputPads() {
  std::lock_guard<std::mutex> l(g_mu);
  Refresh();
  std::vector<XInputPad> pads;
  for (uint32_t i = 0; i < 4; i++) {
    SDL_Gamepad* gp = g_pads[i].gp;
    if (g_pads[i].rx >= 0) {  // as Windows lists the receiver's controllers
      pads.push_back({i, 0x045E, 0x0719, "Controller (Xbox 360 Wireless Receiver for Windows)"});
      continue;
    }
    if (!gp) continue;
    const char* name = SDL_GetGamepadName(gp);
    pads.push_back({i, SDL_GetGamepadVendor(gp), SDL_GetGamepadProduct(gp),
                    name ? name : "Controller"});
  }
  return pads;
}

}  // namespace recomp::hle

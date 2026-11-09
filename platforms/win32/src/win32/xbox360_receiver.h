#pragma once
// The Xbox 360 Wireless Receiver for Windows (and clones of it) read
// directly over libusb (xbox360_receiver.cpp), for hosts with no driver for
// it (macOS). Each of its four slots carries one wireless controller,
// whose input reports are already XINPUT_GAMEPAD; xinput.cpp gives the
// connected slots XInput user indices next to SDL's gamepads. Without
// libusb at build time (RECOMP_X360_RECEIVER undefined) there are no
// slots.

#include <cstdint>

namespace recomp::hle::x360rx {

constexpr int kSlots = 4;

struct SlotState {
  uint8_t subtype = 1;     // XINPUT_DEVSUBTYPE_* from the controller's announcement
  uint8_t gamepad[12]{};   // XINPUT_GAMEPAD
};

// Starts looking for a receiver (once; later calls do nothing). It is
// opened when plugged in, and again after it is unplugged and replugged.
void Start();

// Slot `slot`'s controller; false if none is connected there.
bool Read(int slot, SlotState* out);

// Motor speeds (large, small; 0-255); sent when they change.
void SetRumble(int slot, uint8_t large, uint8_t small);

// Lights the ring quadrant for XInput user index `player` (0-3), or turns
// the slot's light off (-1).
void SetLed(int slot, int player);

}  // namespace recomp::hle::x360rx

// The Xbox 360 Wireless Receiver for Windows over libusb (see the header).
//
// The receiver is a vendor-class USB device (no HID), so a host without a
// driver for it leaves it unclaimed and a user process may claim it. It is
// found by its interfaces rather than its ids, so clones with other
// product ids work: one interface per controller slot with class FF,
// subclass 5D, protocol 81 and an interrupt IN/OUT endpoint pair (protocol
// 82 is the headset's). Packets on a slot (as the Linux and SDL drivers
// read them):
// - 08 xx: presence; bit 7 of xx = controller connected (bit 6 = headset).
// - 00 0F 00 F0 ..., 29 bytes: the controller's announcement; byte 25 bits
//   0-6 are its XInput subtype (1 gamepad, 6/7/0B guitar, 8 drums...).
// - 00 x1 .. .. 00 13 <XINPUT_GAMEPAD>, 29 bytes (bit 0 of byte 1 set):
//   an input report; from byte 6, wButtons, the triggers and the sticks
//   exactly as XInput reports them (stick Y up). XInputGetState leaves out
//   the guide button (0x0400), so it is cleared here.
// Commands are 12 bytes on the slot's OUT endpoint: 08 00 0F C0 asks for
// the presence packets, 00 00 08 4m sets the ring light (m = 6 + quadrant,
// 0 = off), 00 01 0F C0 00 LL SS the motors.
// One thread per slot reads; a supervisor thread opens the receiver
// (rescanning every 2 s while there is none), sends the queued commands and
// closes it when a read fails (unplugged). <ID>_X360RX_TRACE=1 logs every
// packet.

#include "win32/xbox360_receiver.h"

#ifdef RECOMP_X360_RECEIVER

#include <libusb.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#include "runtime/log.h"
#include "x86/runtime.h"

namespace recomp::hle::x360rx {
namespace {

using Command = std::array<uint8_t, 12>;

struct Slot {
  int iface = -1;
  uint8_t ep_in = 0, ep_out = 0;
  bool connected = false;
  SlotState state;
  int rumble = -1;  // last sent (large << 8 | small), -1 none
  int led = -2;     // last sent player, -2 none
};

struct Receiver {
  std::mutex mu;  // everything below
  std::condition_variable cv;
  libusb_context* ctx = nullptr;
  libusb_device_handle* dev = nullptr;
  Slot slots[kSlots];
  int nslots = 0;
  std::deque<std::pair<int, Command>> out;
  std::vector<std::thread> readers;
  std::atomic<bool> lost{false};  // a read failed: close the receiver
  bool trace = false;
};

// Never destroyed: its threads run until the process exits.
Receiver& R() {
  static Receiver* r = new Receiver;
  return *r;
}

void Queue(Receiver& r, int slot, const Command& cmd) {
  r.out.push_back({slot, cmd});
  r.cv.notify_one();
}

void Handle(Receiver& r, int i, const uint8_t* d, int n) {
  if (r.trace) {
    char hex[3 * 32 + 1] = {};
    for (int k = 0; k < n && k < 32; k++) std::snprintf(hex + 3 * k, 4, " %02x", d[k]);
    WarnF("x360rx: slot %d:%s\n", i + 1, hex);
  }
  std::lock_guard<std::mutex> l(r.mu);
  Slot& s = r.slots[i];
  if (n == 2 && d[0] == 0x08) {
    const bool present = (d[1] & 0x80) != 0;
    if (present == s.connected) return;
    s.connected = present;
    s.state = SlotState{};
    s.rumble = -1;
    s.led = -2;
    WarnF("x360rx: slot %d %s\n", i + 1, present ? "connected" : "disconnected");
  } else if (n >= 26 && d[0] == 0x00 && d[1] == 0x0F && d[2] == 0x00 && d[3] == 0xF0) {
    s.state.subtype = d[25] & 0x7F;
    if (!s.state.subtype) s.state.subtype = 1;
    WarnF("x360rx: slot %d controller subtype %u\n", i + 1, s.state.subtype);
  } else if (n >= 18 && d[0] == 0x00 && (d[1] & 0x01) && d[4] == 0x00) {
    if (!s.connected) {  // reports prove presence (a missed presence packet)
      s.connected = true;
      WarnF("x360rx: slot %d connected\n", i + 1);
    }
    std::memcpy(s.state.gamepad, d + 6, 12);
    s.state.gamepad[1] &= ~0x04;  // guide
  }
}

void ReadSlot(int i) {
  Receiver& r = R();
  libusb_device_handle* dev;
  uint8_t ep;
  {
    std::lock_guard<std::mutex> l(r.mu);
    dev = r.dev;
    ep = r.slots[i].ep_in;
  }
  uint8_t buf[32];
  while (!r.lost) {
    int n = 0;
    int rc = libusb_interrupt_transfer(dev, ep, buf, sizeof buf, &n, 250);
    if (rc == LIBUSB_ERROR_TIMEOUT) continue;
    if (rc != 0) {
      WarnF("x360rx: receiver read failed (%s): closing it\n", libusb_error_name(rc));
      std::lock_guard<std::mutex> l(r.mu);
      r.lost = true;
      r.cv.notify_one();
      break;
    }
    Handle(r, i, buf, n);
  }
}

// The slot interfaces of `d` (FF/5D/81 with an interrupt IN and OUT).
int FindSlots(libusb_device* d, Slot* slots) {
  libusb_config_descriptor* cfg = nullptr;
  if (libusb_get_active_config_descriptor(d, &cfg) != 0 &&
      libusb_get_config_descriptor(d, 0, &cfg) != 0)
    return 0;
  int n = 0;
  for (int i = 0; i < cfg->bNumInterfaces && n < kSlots; i++) {
    const libusb_interface& itf = cfg->interface[i];
    if (itf.num_altsetting < 1) continue;
    const libusb_interface_descriptor& alt = itf.altsetting[0];
    if (alt.bInterfaceClass != 0xFF || alt.bInterfaceSubClass != 0x5D ||
        alt.bInterfaceProtocol != 0x81)
      continue;
    Slot s;
    s.iface = alt.bInterfaceNumber;
    for (int e = 0; e < alt.bNumEndpoints; e++) {
      const libusb_endpoint_descriptor& ep = alt.endpoint[e];
      if ((ep.bmAttributes & LIBUSB_TRANSFER_TYPE_MASK) != LIBUSB_TRANSFER_TYPE_INTERRUPT)
        continue;
      if (ep.bEndpointAddress & LIBUSB_ENDPOINT_IN)
        s.ep_in = ep.bEndpointAddress;
      else
        s.ep_out = ep.bEndpointAddress;
    }
    if (s.ep_in && s.ep_out) slots[n++] = s;
  }
  libusb_free_config_descriptor(cfg);
  return n;
}

// Opens the first receiver plugged in; true if one is open.
bool Open(Receiver& r) {
  static bool warned_busy = false;
  libusb_device** list = nullptr;
  ssize_t count = libusb_get_device_list(r.ctx, &list);
  bool opened = false;
  for (ssize_t k = 0; k < count && !opened; k++) {
    Slot slots[kSlots];
    int n = FindSlots(list[k], slots);
    if (!n) continue;
    libusb_device_descriptor dd{};
    libusb_get_device_descriptor(list[k], &dd);
    libusb_device_handle* h = nullptr;
    if (int rc = libusb_open(list[k], &h); rc != 0) {
      WarnF("x360rx: cannot open receiver %04x:%04x: %s\n", dd.idVendor, dd.idProduct,
            libusb_error_name(rc));
      continue;
    }
    int config = 0;
    if (libusb_get_configuration(h, &config) == 0 && config == 0)
      libusb_set_configuration(h, 1);  // left unconfigured with no driver
    int claimed = 0;
    for (; claimed < n; claimed++) {
      int rc = libusb_claim_interface(h, slots[claimed].iface);
      if (rc == 0) continue;
      if (!warned_busy) {
        WarnF("x360rx: receiver %04x:%04x is in use by another program (%s; Steam "
              "takes it while running): retrying\n",
              dd.idVendor, dd.idProduct, libusb_error_name(rc));
        warned_busy = true;
      }
      break;
    }
    if (claimed < n) {
      for (int i = 0; i < claimed; i++) libusb_release_interface(h, slots[i].iface);
      libusb_close(h);
      continue;
    }
    WarnF("x360rx: receiver %04x:%04x, %d controller slots\n", dd.idVendor, dd.idProduct, n);
    warned_busy = false;
    std::lock_guard<std::mutex> l(r.mu);
    r.dev = h;
    r.nslots = n;
    for (int i = 0; i < kSlots; i++) r.slots[i] = i < n ? slots[i] : Slot{};
    r.lost = false;
    for (int i = 0; i < n; i++) Queue(r, i, {0x08, 0x00, 0x0F, 0xC0});
    for (int i = 0; i < n; i++) r.readers.emplace_back(ReadSlot, i);
    opened = true;
  }
  libusb_free_device_list(list, 1);
  return opened;
}

void Close(Receiver& r) {
  std::vector<std::thread> readers;
  {
    std::lock_guard<std::mutex> l(r.mu);
    r.lost = true;
    readers.swap(r.readers);
  }
  for (std::thread& t : readers) t.join();
  std::lock_guard<std::mutex> l(r.mu);
  for (int i = 0; i < r.nslots; i++) libusb_release_interface(r.dev, r.slots[i].iface);
  libusb_close(r.dev);
  r.dev = nullptr;
  r.nslots = 0;
  for (Slot& s : r.slots) s = Slot{};
  r.out.clear();
}

void Supervise() {
  Receiver& r = R();
  for (;;) {
    if (!r.dev && !Open(r)) {
      std::this_thread::sleep_for(std::chrono::seconds(2));
      continue;
    }
    std::deque<std::pair<int, Command>> out;
    {
      std::unique_lock<std::mutex> l(r.mu);
      r.cv.wait_for(l, std::chrono::seconds(1), [&] { return !r.out.empty() || r.lost; });
      if (r.lost) {
        l.unlock();
        Close(r);
        continue;
      }
      out.swap(r.out);
    }
    for (auto& [slot, cmd] : out) {
      int sent = 0;
      int rc = libusb_interrupt_transfer(r.dev, r.slots[slot].ep_out, cmd.data(),
                                         int(cmd.size()), &sent, 100);
      if (rc != 0 && r.trace)
        WarnF("x360rx: slot %d command failed: %s\n", slot + 1, libusb_error_name(rc));
    }
  }
}

}  // namespace

void Start() {
  static std::once_flag once;
  std::call_once(once, [] {
    Receiver& r = R();
    r.trace = Env("X360RX_TRACE") != nullptr;
    if (int rc = libusb_init(&r.ctx); rc != 0) {
      WarnF("x360rx: libusb init failed: %s\n", libusb_error_name(rc));
      return;
    }
    std::thread(Supervise).detach();
  });
}

bool Read(int slot, SlotState* out) {
  if (slot < 0 || slot >= kSlots) return false;
  Receiver& r = R();
  std::lock_guard<std::mutex> l(r.mu);
  const Slot& s = r.slots[slot];
  if (!s.connected) return false;
  if (out) *out = s.state;
  return true;
}

void SetRumble(int slot, uint8_t large, uint8_t small) {
  if (slot < 0 || slot >= kSlots) return;
  Receiver& r = R();
  std::lock_guard<std::mutex> l(r.mu);
  Slot& s = r.slots[slot];
  const int v = large << 8 | small;
  if (!s.connected || s.rumble == v) return;
  s.rumble = v;
  Queue(r, slot, {0x00, 0x01, 0x0F, 0xC0, 0x00, large, small});
}

void SetLed(int slot, int player) {
  if (slot < 0 || slot >= kSlots) return;
  Receiver& r = R();
  std::lock_guard<std::mutex> l(r.mu);
  Slot& s = r.slots[slot];
  if (!s.connected || s.led == player) return;
  s.led = player;
  const uint8_t mode = player >= 0 ? uint8_t(6 + (player & 3)) : 0;
  Queue(r, slot, {0x00, 0x00, 0x08, uint8_t(0x40 + mode)});
}

}  // namespace recomp::hle::x360rx

#else  // no libusb: no receiver

namespace recomp::hle::x360rx {
void Start() {}
bool Read(int, SlotState*) { return false; }
void SetRumble(int, uint8_t, uint8_t) {}
void SetLed(int, int) {}
}  // namespace recomp::hle::x360rx

#endif

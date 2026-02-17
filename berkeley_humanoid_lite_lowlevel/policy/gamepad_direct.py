# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

"""
Gamepad Controller Module for Berkeley Humanoid Lite (Direct Joystick API)

This module implements direct /dev/input/js* reading for gamepad control,
bypassing the inputs library. This works with Nintendo Pro Controller and
other joysticks that expose /dev/input/js* devices.
"""

import struct
import threading
import os
import fcntl
import time
from typing import Dict, Optional


class JoystickEvent:
    """Represents a single joystick event from /dev/input/js*"""
    def __init__(self, timestamp: int, value: int, event_type: int, number: int):
        self.timestamp = timestamp
        self.value = value
        self.type = event_type
        self.number = number
        self.is_button = (event_type & 0x01) != 0
        self.is_axis = (event_type & 0x02) != 0
        self.is_init = (event_type & 0x80) != 0


class XInputEntry:
    """
    Constants for gamepad button and axis mappings.

    This class defines the standard mapping for various gamepad controls,
    including analog sticks, triggers, d-pad, and buttons.

    For Linux joystick API, we use axis/button numbers instead of event codes.
    """
    # Axis mappings (axis numbers for Nintendo Pro Controller)
    AXIS_X_L = 0        # Left stick X
    AXIS_Y_L = 1        # Left stick Y
    AXIS_X_R = 2        # Right stick X
    AXIS_Y_R = 3        # Right stick Y
    AXIS_TRIGGER_L = 4  # Left trigger (may vary)
    AXIS_TRIGGER_R = 5  # Right trigger (may vary)

    # D-pad axes
    BTN_HAT_X = 6       # D-pad X
    BTN_HAT_Y = 7       # D-pad Y

    # Button mappings (button numbers for Nintendo Pro Controller)
    BTN_A = 0           # A button (South)
    BTN_B = 1           # B button (East)
    BTN_X = 3           # X button (North)
    BTN_Y = 4           # Y button (West)
    BTN_BUMPER_L = 6    # L button
    BTN_BUMPER_R = 7    # R button
    BTN_THUMB_L = 13    # Left stick press
    BTN_THUMB_R = 14    # Right stick press
    BTN_BACK = 10       # Minus button
    BTN_START = 11      # Plus button


class Se2Gamepad:
    def __init__(self,
                 stick_sensitivity: float = 1.0,
                 dead_zone: float = 0.01,
                 device_path: str = "/dev/input/js0",
                 ) -> None:
        self.stick_sensitivity = stick_sensitivity
        self.dead_zone = dead_zone
        self.device_path = device_path

        self._stopped = threading.Event()
        self._run_forever_thread = None
        self._gamepad_available = False
        self._gamepad_error_shown = False
        self._js_file: Optional[any] = None

        # State storage: axes and buttons by number
        self._axis_states = {}
        self._button_states = {}

        self.commands = {
            "velocity_x": 0.0,
            "velocity_y": 0.0,
            "velocity_yaw": 0.0,
            "mode_switch": 0,
            "squat_depth": 0.0,  # 0.0=standing, 1.0=full squat (Right Stick Y)
        }

        self._open_joystick()

    def _open_joystick(self) -> bool:
        """Try to open the joystick device."""
        try:
            if os.path.exists(self.device_path):
                self._js_file = open(self.device_path, 'rb')
                # Set non-blocking mode
                fd = self._js_file.fileno()
                flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

                self._gamepad_available = True
                print(f"Gamepad connected: {self.device_path}")
                return True
            else:
                if not self._gamepad_error_shown:
                    print(f"Gamepad device not found: {self.device_path}")
                    self._gamepad_error_shown = True
                return False
        except PermissionError:
            if not self._gamepad_error_shown:
                print(f"Permission denied accessing {self.device_path}")
                print("To enable gamepad support:")
                print("  1. Add your user to the input group: sudo usermod -a -G input $USER")
                print("  2. Log out and log back in")
                print("  3. Verify access: ls -la /dev/input/js0")
                self._gamepad_error_shown = True
            return False
        except Exception as e:
            if not self._gamepad_error_shown:
                print(f"Error opening gamepad: {e}")
                self._gamepad_error_shown = True
            return False

    def reset(self) -> None:
        self._axis_states = {}
        self._button_states = {}

    def stop(self) -> None:
        print("Gamepad stopping...")
        self._stopped.set()
        if self._js_file:
            try:
                self._js_file.close()
            except:
                pass

    def run(self) -> None:
        self._run_forever_thread = threading.Thread(target=self.run_forever, daemon=True)
        self._run_forever_thread.start()

    def run_forever(self) -> None:
        while not self._stopped.is_set():
            self.advance()
            time.sleep(0.001)  # Small sleep to prevent CPU spinning

    def _read_event(self) -> Optional[JoystickEvent]:
        """Read a single event from the joystick device.

        Event format: struct js_event {
            __u32 time;     /* event timestamp in milliseconds */
            __s16 value;    /* value */
            __u8 type;      /* event type */
            __u8 number;    /* axis/button number */
        };
        """
        if not self._js_file:
            return None

        try:
            # Read 8 bytes (event structure)
            data = self._js_file.read(8)
            if not data or len(data) < 8:
                return None

            # Unpack: I=unsigned int, h=short, 2B=2 unsigned chars
            timestamp, value, event_type, number = struct.unpack('Ihbb', data)
            return JoystickEvent(timestamp, value, event_type, number)
        except BlockingIOError:
            # No data available (non-blocking mode)
            return None
        except Exception:
            return None

    def advance(self) -> None:
        if not self._gamepad_available:
            # Try to reconnect
            if not self._open_joystick():
                return

        try:
            # Read all available events (non-blocking by reading until no more data)
            while True:
                event = self._read_event()
                if event is None:
                    break

                # Skip init events
                if event.is_init:
                    continue

                # Update state
                if event.is_axis:
                    self._axis_states[event.number] = event.value
                elif event.is_button:
                    self._button_states[event.number] = event.value

            self._update_command_buffer()

        except Exception as e:
            if not self._gamepad_error_shown:
                print(f"Error reading gamepad: {e}")
                print("\nContinuing without gamepad - using default commands...")
                self._gamepad_error_shown = True
            self._gamepad_available = False
            if self._js_file:
                try:
                    self._js_file.close()
                except:
                    pass
                self._js_file = None

    def _update_command_buffer(self) -> Dict[str, float]:
        # Get axis values (joystick axes range from -32767 to 32767)
        velocity_x = self._axis_states.get(XInputEntry.AXIS_Y_L, 0)
        velocity_y = self._axis_states.get(XInputEntry.AXIS_X_R, 0)
        velocity_yaw = self._axis_states.get(XInputEntry.AXIS_X_L, 0)

        # Normalize to -1.0 to 1.0
        self.commands["velocity_x"] = velocity_x / -32767.0
        self.commands["velocity_y"] = velocity_y / -32767.0
        self.commands["velocity_yaw"] = velocity_yaw / -32767.0

        # Right Stick Y for squat depth (forward = stand, back = squat)
        # Pushing stick back (positive value) = squat, forward (negative) = stand
        squat_raw = self._axis_states.get(XInputEntry.AXIS_Y_R, 0)
        # Normalize: -32767 (forward) -> 0.0, +32767 (back) -> 1.0
        squat_normalized = (squat_raw + 32767) / 65534.0
        # Apply dead zone
        if squat_normalized < self.dead_zone:
            squat_normalized = 0.0
        self.commands["squat_depth"] = squat_normalized

        mode_switch = 0

        # Get button states (buttons are 0 or 1)
        btn_a = self._button_states.get(XInputEntry.BTN_A, 0)
        btn_x = self._button_states.get(XInputEntry.BTN_X, 0)
        btn_bumper_l = self._button_states.get(XInputEntry.BTN_BUMPER_L, 0)
        btn_bumper_r = self._button_states.get(XInputEntry.BTN_BUMPER_R, 0)
        btn_thumb_l = self._button_states.get(XInputEntry.BTN_THUMB_L, 0)
        btn_thumb_r = self._button_states.get(XInputEntry.BTN_THUMB_R, 0)

        # Enter RL control mode (A + Right Bumper)
        if btn_a and btn_bumper_r:
            mode_switch = 3

        # Enter init mode (A + Left Bumper)
        if btn_a and btn_bumper_l:
            mode_switch = 2

        # Enter idle mode (X or Left/Right Thumbstick press)
        if btn_x or btn_thumb_l or btn_thumb_r:
            mode_switch = 1

        self.commands["mode_switch"] = mode_switch


if __name__ == "__main__":
    command_controller = Se2Gamepad()
    command_controller.run()

    try:
        while True:
            print(f"""vx: {command_controller.commands.get("velocity_x"):.2f}, vy: {command_controller.commands.get("velocity_y"):.2f}, vyaw: {command_controller.commands.get("velocity_yaw"):.2f}, squat: {command_controller.commands.get("squat_depth"):.2f}, mode: {command_controller.commands.get("mode_switch")}""")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Keyboard interrupt")

    command_controller.stop()

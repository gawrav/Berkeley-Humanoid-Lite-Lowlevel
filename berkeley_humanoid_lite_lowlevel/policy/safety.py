# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

"""Safety shim between policy output and robot actuators."""

import numpy as np


# URDF joint position limits (robot space, radians)
JOINT_POSITION_LOWER = np.array([
    -0.175, -0.982, -1.898,  0.0, -0.785, -0.262,  # left leg
    -1.571, -0.589, -1.898,  0.0, -0.785, -0.262,  # right leg
], dtype=np.float32)

JOINT_POSITION_UPPER = np.array([
     1.571,  0.589,  0.982,  2.443,  0.785,  0.262,  # left leg
     0.175,  0.982,  0.982,  2.443,  0.785,  0.262,  # right leg
], dtype=np.float32)


class SafetyShim:
    """Safety layer between policy output and robot actuators.

    Always-on checks applied to policy actions before they reach the motors:
      1. NaN/Inf detection — substitutes last safe actions
      2. Joint velocity limit — max change between consecutive commands
      3. Joint acceleration limit — max change in commanded velocity between steps
      4. URDF joint position limits — hard clamp on final output
      5. Emergency stop — triggers after N consecutive violations

    Velocity and acceleration are based on consecutive policy outputs (not target vs
    measured position), so the shim limits how fast commands change without fighting
    the PD controller's tracking loop.
    """

    def __init__(self, default_positions, joint_limits_lower, joint_limits_upper,
                 dt, max_joint_velocity=20.0, max_joint_acceleration=200.0,
                 max_consecutive_violations=20):
        self.default_positions = default_positions.copy()
        self.joint_limits_lower = joint_limits_lower
        self.joint_limits_upper = joint_limits_upper
        self.dt = dt
        self.max_joint_velocity = max_joint_velocity      # rad/s
        self.max_joint_acceleration = max_joint_acceleration  # rad/s²
        self.max_violations = max_consecutive_violations

        # Derived per-step limits
        self.max_delta = max_joint_velocity * dt           # rad/step
        self.max_delta_change = max_joint_acceleration * dt * dt  # rad/step²

        self.prev_actions = default_positions.copy()
        self.prev_command_delta = np.zeros_like(default_positions)
        self.last_safe_actions = default_positions.copy()
        self.consecutive_violations = 0
        self.total_violations = 0
        self.emergency_stop = False

    def check(self, actions):
        """Validate and clamp actions. Returns (safe_actions, was_modified)."""
        severe = False  # NaN, velocity, or acceleration violation

        # 1. NaN/Inf check
        if not np.all(np.isfinite(actions)):
            bad = np.where(~np.isfinite(actions))[0]
            print(f"\n  [SAFETY] NaN/Inf in actions at joints {bad.tolist()}, using last safe")
            actions = self.last_safe_actions.copy()
            severe = True

        # 2. Joint velocity limit (max change between consecutive commands)
        command_delta = actions - self.prev_actions
        clamped_delta = np.clip(command_delta, -self.max_delta, self.max_delta)
        if not np.array_equal(clamped_delta, command_delta):
            violations = np.where(clamped_delta != command_delta)[0]
            max_vel = np.max(np.abs(command_delta)) / self.dt
            print(f"\n  [SAFETY] Velocity limit on joints {violations.tolist()}"
                  f" (peak: {max_vel:.1f} rad/s, limit: {self.max_joint_velocity:.1f} rad/s)")
            command_delta = clamped_delta
            severe = True

        # 3. Joint acceleration limit (max change in commanded velocity)
        delta_change = command_delta - self.prev_command_delta
        clamped_change = np.clip(delta_change, -self.max_delta_change, self.max_delta_change)
        if not np.array_equal(clamped_change, delta_change):
            violations = np.where(clamped_change != delta_change)[0]
            max_accel = np.max(np.abs(delta_change)) / (self.dt * self.dt)
            print(f"\n  [SAFETY] Acceleration limit on joints {violations.tolist()}"
                  f" (peak: {max_accel:.0f} rad/s², limit: {self.max_joint_acceleration:.0f} rad/s²)")
            command_delta = self.prev_command_delta + clamped_change
            severe = True

        actions = self.prev_actions + command_delta

        # 4. URDF joint position limits — hard clamp on final output
        # Position clamping is normal near joint limits and does not count
        # toward emergency stop (only NaN/velocity/acceleration do).
        clamped = np.clip(actions, self.joint_limits_lower, self.joint_limits_upper)
        if not np.array_equal(clamped, actions):
            violations = np.where(clamped != actions)[0]
            print(f"\n  [SAFETY] Position limit clamp on joints {violations.tolist()}")
            actions = clamped
        self.prev_actions = actions.copy()
        self.prev_command_delta = command_delta.copy()

        # Track consecutive severe violations for emergency stop
        if severe:
            self.consecutive_violations += 1
            self.total_violations += 1
        else:
            self.consecutive_violations = 0

        if self.consecutive_violations >= self.max_violations:
            print(f"\n  [SAFETY] EMERGENCY STOP: {self.max_violations} consecutive violations")
            self.emergency_stop = True

        self.last_safe_actions = actions.copy()
        return actions, severe

# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

"""
Locomotion Script

Usage:
    uv run scripts/run_locomotion.py                    # Normal mode
    uv run scripts/run_locomotion.py --config configs/policy_biped_50hz.yaml
    uv run scripts/run_locomotion.py --max-steps 500 --log run_001.json
"""

import sys
import argparse
import json
import time
from datetime import datetime
import numpy as np

from cc.udp import UDP
from loop_rate_limiters import RateLimiter

from berkeley_humanoid_lite_lowlevel.robot import Humanoid, State
from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController
from berkeley_humanoid_lite_lowlevel.policy.config import Cfg


JOINT_NAMES = [
    "L_hip_roll",
    "L_hip_yaw",
    "L_hip_pitch",
    "L_knee",
    "L_ankle_pitch",
    "L_ankle_roll",
    "R_hip_roll",
    "R_hip_yaw",
    "R_hip_pitch",
    "R_knee",
    "R_ankle_pitch",
    "R_ankle_roll",
]


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run locomotion policy')
    parser.add_argument('--max-steps', type=int, default=0,
                        help='Max steps in RL_RUNNING mode before stopping (0=unlimited)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run policy but do not apply actions to robot')
    parser.add_argument('--log', type=str, default=None,
                        help='Log data to file (e.g., --log run_001.json)')
    parser.add_argument('--skip-init-position', action='store_true',
                        help='Skip RL_INIT interpolation, keep current position and go straight to RL_RUNNING')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    args, remaining = parser.parse_known_args()

    # Reconstruct sys.argv for Cfg.from_arguments()
    if args.config:
        sys.argv = [sys.argv[0], '--config', args.config] + remaining
    else:
        sys.argv = [sys.argv[0]] + remaining

    # Load configuration
    cfg = Cfg.from_arguments()

    print(f"\nPolicy frequency: {1 / cfg.policy_dt} Hz")
    if args.max_steps > 0:
        print(f"Max RL_RUNNING steps: {args.max_steps}")
    if args.dry_run:
        print("DRY RUN MODE: Policy will execute but actions NOT applied to robot")
    if args.skip_init_position:
        print("SKIP INIT POSITION: Will hold current position and go straight to RL_RUNNING")
    if args.log:
        print(f"Logging to: {args.log}")
    print()

    # Initialize data log
    data_log = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "config_file": args.config,
            "policy_dt": cfg.policy_dt,
            "max_steps": args.max_steps,
            "dry_run": args.dry_run,
            "joint_names": JOINT_NAMES,
        },
        "frames": []
    }

    udp = UDP(("0.0.0.0", 11000), ("192.168.86.40", 11000))

    # Initialize and start policy controller
    controller = RlController(cfg)
    controller.load_policy()

    rate = RateLimiter(1 / cfg.policy_dt)

    robot = Humanoid(skip_init_position=args.skip_init_position)

    robot.enter_damping()

    obs = robot.reset()

    running_step_count = 0
    start_time = None

    try:
        while True:
            # Reset debug counter when first entering RL_RUNNING (before update)
            if robot.state == State.RL_RUNNING and running_step_count == 0:
                controller.reset_debug_counter()
                start_time = time.time()

            actions = controller.update(obs)

            # Check max steps limit for RL_RUNNING
            if robot.state == State.RL_RUNNING:
                running_step_count += 1
                if args.max_steps > 0 and running_step_count > args.max_steps:
                    print(f"\nMax RL_RUNNING steps ({args.max_steps}) reached. Stopping.")
                    break

            # Log data during RL_RUNNING
            if args.log and robot.state == State.RL_RUNNING:
                # Get exact policy inputs/outputs from controller
                policy_data = controller.get_last_frame_data()
                frame = {
                    "step": running_step_count,
                    "timestamp": time.time() - start_time,
                    # Exact policy inputs
                    "policy_input": policy_data["policy_input"],
                    # Exact policy outputs
                    "policy_output": policy_data["policy_output"],
                    # Raw observations (before processing)
                    "raw_observations": policy_data["raw_observations"],
                    # Gamepad commands
                    "gamepad": {
                        "mode_switch": float(obs[31]),
                        "velocity_x": float(obs[32]),
                        "velocity_y": float(obs[33]),
                        "velocity_yaw": float(obs[34]),
                    },
                    # Actual robot state
                    "robot_measured": {
                        "joint_positions": robot.joint_position_measured.tolist(),
                        "joint_velocities": robot.joint_velocity_measured.tolist(),
                    }
                }
                data_log["frames"].append(frame)

            # In dry-run mode, don't apply policy actions - hold current position instead
            if args.dry_run:
                obs = robot.step(robot.joint_position_measured.copy())
            else:
                obs = robot.step(actions)
            udp.send_numpy(obs)

            rate.sleep()

    except KeyboardInterrupt:
        pass

    robot.stop()
    print("\nStopped.")

    # Save logged data
    if args.log and data_log["frames"]:
        data_log["metadata"]["total_frames"] = len(data_log["frames"])
        data_log["metadata"]["duration_seconds"] = data_log["frames"][-1]["timestamp"] if data_log["frames"] else 0
        with open(args.log, 'w') as f:
            json.dump(data_log, f, indent=2)
        print(f"Saved {len(data_log['frames'])} frames to {args.log}")


if __name__ == "__main__":
    main()

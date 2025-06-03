#!/usr/bin/env python3
import math
import time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from TrapezoidProfile import TrapezoidProfileConstraints
from ProfiledPIDController import ProfiledPIDController


@dataclass
class RobotConfig:
    """Configuration parameters for the differential drive robot."""
    wheelbase_inches: float = 20.0  # Distance between wheels
    wheel_diameter_inches: float = 4.0  # Wheel diameter
    max_wheel_rpm: float = 600.0  # Maximum wheel speed

    # Computed properties
    @property
    def wheelbase_meters(self) -> float:
        return self.wheelbase_inches * 0.0254

    @property
    def wheel_radius_meters(self) -> float:
        return (self.wheel_diameter_inches * 0.0254) / 2.0

    @property
    def max_wheel_speed_ms(self) -> float:
        """Maximum wheel speed in m/s"""
        return (self.max_wheel_rpm / 60.0) * 2 * math.pi * self.wheel_radius_meters


@dataclass
class Pose2D:
    """2D robot pose"""
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0  # radians

    def distance_to(self, other: 'Pose2D') -> float:
        """Calculate Euclidean distance to another pose"""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def angle_to(self, other: 'Pose2D') -> float:
        """Calculate angle to another pose"""
        return math.atan2(other.y - self.y, other.x - self.x)


@dataclass
class WheelSpeeds:
    """Wheel speeds for differential drive"""
    left: float = 0.0  # m/s
    right: float = 0.0  # m/s

    def to_motor_commands(self, config: RobotConfig) -> Tuple[float, float]:
        """Convert wheel speeds to motor commands (-1 to 1)"""
        max_speed = config.max_wheel_speed_ms
        left_cmd = np.clip(self.left / max_speed, -1.0, 1.0)
        right_cmd = np.clip(self.right / max_speed, -1.0, 1.0)
        return left_cmd, right_cmd


@dataclass
class Twist2D:
    """2D velocity (linear and angular)"""
    linear: float = 0.0  # m/s
    angular: float = 0.0  # rad/s


class DifferentialDriveKinematics:
    """Handles conversion between robot velocities and wheel speeds"""

    def __init__(self, config: RobotConfig):
        self.config = config

    def to_wheel_speeds(self, twist: Twist2D) -> WheelSpeeds:
        """Convert robot twist to wheel speeds"""
        # v_left = v_linear - (w_angular * wheelbase) / 2
        # v_right = v_linear + (w_angular * wheelbase) / 2
        half_wheelbase = self.config.wheelbase_meters / 2.0

        left_speed = twist.linear - (twist.angular * half_wheelbase)
        right_speed = twist.linear + (twist.angular * half_wheelbase)

        return WheelSpeeds(left_speed, right_speed)

    def from_wheel_speeds(self, wheel_speeds: WheelSpeeds) -> Twist2D:
        """Convert wheel speeds to robot twist"""
        # v_linear = (v_left + v_right) / 2
        # w_angular = (v_right - v_left) / wheelbase
        linear = (wheel_speeds.left + wheel_speeds.right) / 2.0
        angular = (wheel_speeds.right - wheel_speeds.left) / self.config.wheelbase_meters

        return Twist2D(linear, angular)


class DifferentialDriveController:
    """
    High-level controller for differential drive robot navigation.
    Uses profiled PID controllers for smooth motion to target poses.
    """

    def __init__(self, config: RobotConfig,
                 linear_constraints: Optional[TrapezoidProfileConstraints] = None,
                 angular_constraints: Optional[TrapezoidProfileConstraints] = None,
                 control_period: float = 0.02):
        """
        Initialize the differential drive controller.

        Args:
            config: Robot configuration
            linear_constraints: Velocity/acceleration limits for linear motion
            angular_constraints: Velocity/acceleration limits for angular motion
            control_period: Control loop period in seconds
        """
        self.config = config
        self.kinematics = DifferentialDriveKinematics(config)
        self.control_period = control_period

        # Default motion constraints if not provided
        if linear_constraints is None:
            max_linear_vel = config.max_wheel_speed_ms * 0.8  # 80% of max for safety
            linear_constraints = TrapezoidProfileConstraints(
                max_velocity=max_linear_vel,
                max_acceleration=max_linear_vel * 2.0  # Reach max speed in 0.5s
            )

        if angular_constraints is None:
            # Maximum angular velocity when one wheel forward, one backward at max speed
            max_angular_vel = (2 * config.max_wheel_speed_ms) / config.wheelbase_meters
            max_angular_vel *= 0.6  # 60% of theoretical max for safety
            angular_constraints = TrapezoidProfileConstraints(
                max_velocity=max_angular_vel,
                max_acceleration=max_angular_vel * 4.0  # Reach max rotation in 0.25s
            )

        # PID Controllers for linear and angular motion
        # You may need to tune these gains for your specific robot
        self.linear_controller = ProfiledPIDController(
            kp=2.0, ki=0.0, kd=0.1,
            constraints=linear_constraints,
            period=control_period
        )

        self.angular_controller = ProfiledPIDController(
            kp=3.0, ki=0.0, kd=0.15,
            constraints=angular_constraints,
            period=control_period
        )

        # Enable continuous input for angular controller (handles angle wraparound)
        self.angular_controller.enable_continuous_input(-math.pi, math.pi)

        # Set tolerances
        self.linear_controller.set_tolerance(0.05)  # 5cm position tolerance
        self.angular_controller.set_tolerance(math.radians(5))  # 5 degree angle tolerance

        # Current state
        self.current_pose = Pose2D()
        self.target_pose = Pose2D()
        self.enabled = False

        # Thread safety
        self.lock = threading.Lock()

    def set_current_pose(self, pose: Pose2D):
        """Update the current robot pose (called from vision system)"""
        with self.lock:
            self.current_pose = pose

    def set_target_pose(self, target: Pose2D):
        """Set a new target pose for the robot"""
        with self.lock:
            self.target_pose = target
            # Reset controllers when setting new target
            self.linear_controller.reset(0.0)  # Reset to current distance (0)
            self.angular_controller.reset(self.current_pose.yaw)

    def enable(self):
        """Enable the controller"""
        self.enabled = True

    def disable(self):
        """Disable the controller"""
        self.enabled = False

    def at_target(self) -> bool:
        """Check if robot has reached the target pose"""
        if not self.enabled:
            return False

        with self.lock:
            # Check both linear and angular goals
            distance_to_target = self.current_pose.distance_to(self.target_pose)
            angle_error = abs(self._normalize_angle(self.target_pose.yaw - self.current_pose.yaw))

            position_ok = distance_to_target <= 0.05  # 5cm
            angle_ok = angle_error <= math.radians(5)  # 5 degrees

            return position_ok and angle_ok

    def calculate_control_output(self) -> Tuple[float, float]:
        """
        Calculate motor commands to reach the target pose.

        Returns:
            Tuple of (left_motor_command, right_motor_command) in range [-1, 1]
        """
        if not self.enabled:
            return (0.0, 0.0)

        with self.lock:
            current = self.current_pose
            target = self.target_pose

        # Calculate distance and angle to target
        distance_to_target = current.distance_to(target)
        angle_to_target = current.angle_to(target)

        # Calculate desired heading (direction to target)
        heading_error = self._normalize_angle(angle_to_target - current.yaw)

        # Strategy: First orient towards target, then drive while fine-tuning orientation

        # Linear velocity based on distance to target
        linear_output = self.linear_controller.calculate(0.0, distance_to_target)

        # Reduce linear speed when we need to turn significantly
        if abs(heading_error) > math.radians(30):  # If heading error > 30 degrees
            linear_output *= 0.3  # Reduce linear speed significantly
        elif abs(heading_error) > math.radians(15):  # If heading error > 15 degrees
            linear_output *= 0.6  # Reduce linear speed moderately

        # Angular velocity to reach target orientation
        # When far from target, prioritize heading towards target
        # When close to target, prioritize final orientation
        if distance_to_target > 0.2:  # Far from target - focus on heading
            desired_yaw = angle_to_target
        else:  # Close to target - focus on final orientation
            desired_yaw = target.yaw

        angular_output = self.angular_controller.calculate(current.yaw, desired_yaw)

        # Create twist and convert to wheel speeds
        twist = Twist2D(linear=linear_output, angular=angular_output)
        wheel_speeds = self.kinematics.to_wheel_speeds(twist)

        # Convert to motor commands
        return wheel_speeds.to_motor_commands(self.config)

    def get_status(self) -> dict:
        """Get controller status information"""
        with self.lock:
            current = self.current_pose
            target = self.target_pose

        distance_to_target = current.distance_to(target)
        angle_error = self._normalize_angle(target.yaw - current.yaw)

        return {
            'enabled': self.enabled,
            'current_pose': current,
            'target_pose': target,
            'distance_to_target': distance_to_target,
            'angle_error_deg': math.degrees(angle_error),
            'at_target': self.at_target(),
            'linear_at_setpoint': self.linear_controller.at_setpoint(),
            'angular_at_setpoint': self.angular_controller.at_setpoint()
        }

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


class RobotNavigationSystem:
    """
    Complete navigation system that integrates with your vision system.
    """

    def __init__(self, config: RobotConfig):
        self.config = config
        self.controller = DifferentialDriveController(config)

        # For sending motor commands (you'll need to implement this)
        self.motor_command_callback = None

        # Control loop
        self.control_loop_running = False
        self.control_thread = None

    def set_motor_command_callback(self, callback):
        """Set callback function to send motor commands to hardware"""
        self.motor_command_callback = callback

    def update_pose_from_vision(self, x: float, y: float, yaw: float):
        """Update robot pose from vision system (call this when you get fused pose)"""
        pose = Pose2D(x, y, yaw)
        self.controller.set_current_pose(pose)

    def navigate_to(self, x: float, y: float, yaw: float):
        """Command robot to navigate to target pose"""
        target = Pose2D(x, y, yaw)
        self.controller.set_target_pose(target)
        self.controller.enable()

        # Start control loop if not running
        if not self.control_loop_running:
            self.start_control_loop()

    def stop(self):
        """Stop the robot"""
        self.controller.disable()
        if self.motor_command_callback:
            self.motor_command_callback(0.0, 0.0)

    def start_control_loop(self):
        """Start the control loop in a separate thread"""
        if self.control_loop_running:
            return

        self.control_loop_running = True
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()

    def stop_control_loop(self):
        """Stop the control loop"""
        self.control_loop_running = False
        if self.control_thread:
            self.control_thread.join()

    def _control_loop(self):
        """Main control loop (runs in separate thread)"""
        while self.control_loop_running:
            start_time = time.time()

            # Calculate control output
            left_cmd, right_cmd = self.controller.calculate_control_output()

            # Send motor commands
            if self.motor_command_callback:
                self.motor_command_callback(left_cmd, right_cmd)

            # Sleep to maintain control rate
            elapsed = time.time() - start_time
            sleep_time = max(0, self.controller.control_period - elapsed)
            time.sleep(sleep_time)

    def get_status(self) -> dict:
        """Get navigation system status"""
        return self.controller.get_status()

    def is_at_target(self) -> bool:
        """Check if robot has reached target"""
        return self.controller.at_target()


# Example integration with your vision system
class VisionIntegratedRobot:
    """Example of how to integrate with your existing vision system"""

    def __init__(self):
        # Configure your robot parameters
        self.robot_config = RobotConfig(
            wheelbase_inches=20.0,
            wheel_diameter_inches=4.0,
            max_wheel_rpm=600.0
        )

        # Create navigation system
        self.nav_system = RobotNavigationSystem(self.robot_config)

        # Set up motor command callback
        self.nav_system.set_motor_command_callback(self.send_motor_commands)

    def send_motor_commands(self, left_cmd: float, right_cmd: float):
        """
        Send motor commands to your robot hardware.
        You'll need to implement this based on your communication protocol.
        """
        print(f"Motor Commands: Left={left_cmd:.3f}, Right={right_cmd:.3f}")
        # TODO: Send commands to your robot (via serial, TCP, etc.)

    def process_vision_update(self, x: float, y: float, yaw: float):
        """
        Call this function when you get a pose update from your vision system.
        This would replace/augment the pose handling in your main vision loop.
        """
        # Update robot pose
        self.nav_system.update_pose_from_vision(x, y, yaw)

        # Print status
        status = self.nav_system.get_status()
        if status['enabled']:
            print(f"Nav Status: Distance to target: {status['distance_to_target']:.3f}m, "
                  f"Angle error: {status['angle_error_deg']:.1f}°, "
                  f"At target: {status['at_target']}")

    def navigate_to_pose(self, x: float, y: float, yaw_degrees: float):
        """Navigate to a target pose"""
        yaw_radians = math.radians(yaw_degrees)
        print(f"Navigating to: x={x:.3f}, y={y:.3f}, yaw={yaw_degrees:.1f}°")
        self.nav_system.navigate_to(x, y, yaw_radians)

    def stop_navigation(self):
        """Stop navigation"""
        self.nav_system.stop()


# Example usage
if __name__ == "__main__":
    # Create robot instance
    robot = VisionIntegratedRobot()

    # Simulate receiving pose updates from vision system
    print("=== Simulating Vision-Based Navigation ===")

    # Set initial pose
    robot.process_vision_update(0.0, 0.0, 0.0)

    # Navigate to target
    robot.navigate_to_pose(1.0, 1.0, 90.0)  # Go to (1m, 1m) facing 90 degrees

    # Simulate vision updates over time
    for i in range(100):
        # Simulate robot movement (this would come from your actual vision system)
        progress = i / 100.0
        current_x = progress * 1.0
        current_y = progress * 1.0
        current_yaw = progress * math.pi / 2

        robot.process_vision_update(current_x, current_y, current_yaw)

        if robot.nav_system.is_at_target():
            print("Target reached!")
            break

        time.sleep(0.1)

    # Stop navigation
    robot.stop_navigation()
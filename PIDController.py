import time
import math
from typing import Optional, Tuple
from dataclasses import dataclass


class PIDController:
    """
    A basic PID controller implementation.
    """

    def __init__(self, kp: float, ki: float, kd: float, period: float = 0.02):
        """
        Initialize PID controller.

        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            period: Control loop period in seconds
        """
        if kp < 0 or ki < 0 or kd < 0:
            raise ValueError("PID gains must be non-negative")
        if period <= 0:
            raise ValueError("Period must be positive")

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.period = period

        # Internal state
        self.previous_error = 0.0
        self.accumulated_error = 0.0
        self.setpoint = 0.0

        # Constraints
        self.min_integral = float('-inf')
        self.max_integral = float('inf')
        self.izone = float('inf')

        # Tolerance
        self.position_tolerance = 0.05
        self.velocity_tolerance = float('inf')

        # Continuous input
        self.continuous_input_enabled = False
        self.min_input = 0.0
        self.max_input = 0.0

        self.first_run = True

    def set_pid(self, kp: float, ki: float, kd: float):
        """Set PID gains."""
        if kp < 0 or ki < 0 or kd < 0:
            raise ValueError("PID gains must be non-negative")
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def set_setpoint(self, setpoint: float):
        """Set the controller setpoint."""
        self.setpoint = setpoint

    def set_tolerance(self, position_tolerance: float, velocity_tolerance: float = float('inf')):
        """Set position and velocity tolerances for at_setpoint()."""
        self.position_tolerance = abs(position_tolerance)
        self.velocity_tolerance = abs(velocity_tolerance)

    def set_integrator_range(self, min_integral: float, max_integral: float):
        """Set limits on integral accumulation to prevent windup."""
        self.min_integral = min_integral
        self.max_integral = max_integral

    def set_izone(self, izone: float):
        """Set integral zone - integral only accumulates when error is within this range."""
        if izone < 0:
            raise ValueError("IZone must be non-negative")
        self.izone = izone

    def enable_continuous_input(self, min_input: float, max_input: float):
        """Enable continuous input (e.g., for angles that wrap around)."""
        self.continuous_input_enabled = True
        self.min_input = min_input
        self.max_input = max_input

    def disable_continuous_input(self):
        """Disable continuous input."""
        self.continuous_input_enabled = False

    def calculate(self, measurement: float, setpoint: Optional[float] = None) -> float:
        """
        Calculate PID output.

        Args:
            measurement: Current process variable measurement
            setpoint: Optional new setpoint

        Returns:
            Control output
        """
        if setpoint is not None:
            self.setpoint = setpoint

        # Calculate error
        error = self.setpoint - measurement

        # Handle continuous input
        if self.continuous_input_enabled:
            input_range = self.max_input - self.min_input
            error = self._input_modulus(error, -input_range / 2, input_range / 2)

        # Integral term with windup protection
        if abs(error) <= self.izone:
            self.accumulated_error += error * self.period
            self.accumulated_error = max(min(self.accumulated_error, self.max_integral), self.min_integral)
        else:
            self.accumulated_error = 0.0

        # Derivative term
        if self.first_run:
            derivative = 0.0
            self.first_run = False
        else:
            derivative = (error - self.previous_error) / self.period

        # Calculate output
        output = (self.kp * error +
                  self.ki * self.accumulated_error +
                  self.kd * derivative)

        self.previous_error = error
        return output

    def at_setpoint(self) -> bool:
        """Check if the controller has reached the setpoint within tolerance."""
        if self.first_run:
            return False

        position_error = abs(self.previous_error)
        velocity_error = abs(self.previous_error - self.accumulated_error) / self.period if self.period > 0 else 0

        return (position_error <= self.position_tolerance and
                velocity_error <= self.velocity_tolerance)

    def reset(self):
        """Reset the controller state."""
        self.previous_error = 0.0
        self.accumulated_error = 0.0
        self.first_run = True

    def get_error(self) -> float:
        """Get the current error."""
        return self.previous_error

    def get_accumulated_error(self) -> float:
        """Get the accumulated integral error."""
        return self.accumulated_error

    @staticmethod
    def _input_modulus(input_val: float, min_input: float, max_input: float) -> float:
        """Calculate input modulus for continuous input."""
        modulus = max_input - min_input
        num_max = int((input_val - min_input) / modulus)
        return input_val - num_max * modulus


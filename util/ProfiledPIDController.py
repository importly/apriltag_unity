from typing import Optional

from util.TrapezoidProfile import TrapezoidProfile, TrapezoidProfileState, TrapezoidProfileConstraints
from util.PIDController import PIDController


class ProfiledPIDController:
    """
    PID Controller with trapezoidal motion profiling.
    Constrains the setpoint to follow smooth acceleration/deceleration profiles.
    """

    def __init__(self, kp: float, ki: float, kd: float,
                 constraints: TrapezoidProfileConstraints, period: float = 0.02):
        """
        Initialize profiled PID controller.

        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            constraints: Velocity and acceleration constraints
            period: Control loop period in seconds
        """
        self.controller = PIDController(kp, ki, kd, period)
        self.constraints = constraints
        self.profile = TrapezoidProfile(constraints)

        self.goal = TrapezoidProfileState()
        self.setpoint = TrapezoidProfileState()

        # For continuous input
        self.min_input = 0.0
        self.max_input = 0.0

    def set_pid(self, kp: float, ki: float, kd: float):
        """Set PID gains."""
        self.controller.set_pid(kp, ki, kd)

    def set_goal(self, goal):
        """Set the goal state or position."""
        if isinstance(goal, (int, float)):
            self.goal = TrapezoidProfileState(float(goal), 0.0)
        elif isinstance(goal, TrapezoidProfileState):
            self.goal = goal
        else:
            raise TypeError("Goal must be a number or TrapezoidProfileState")

    def set_constraints(self, constraints: TrapezoidProfileConstraints):
        """Set velocity and acceleration constraints."""
        self.constraints = constraints
        self.profile = TrapezoidProfile(constraints)

    def enable_continuous_input(self, min_input: float, max_input: float):
        """Enable continuous input."""
        self.controller.enable_continuous_input(min_input, max_input)
        self.min_input = min_input
        self.max_input = max_input

    def disable_continuous_input(self):
        """Disable continuous input."""
        self.controller.disable_continuous_input()

    def set_tolerance(self, position_tolerance: float, velocity_tolerance: float = float('inf')):
        """Set tolerances for goal checking."""
        self.controller.set_tolerance(position_tolerance, velocity_tolerance)

    def calculate(self, measurement: float, goal=None,
                  constraints: Optional[TrapezoidProfileConstraints] = None) -> float:
        """
        Calculate the next control output.

        Args:
            measurement: Current measurement
            goal: Optional new goal (position or state)
            constraints: Optional new constraints

        Returns:
            Control output
        """
        if goal is not None:
            self.set_goal(goal)

        if constraints is not None:
            self.set_constraints(constraints)

        # Handle continuous input for profiling
        if self.controller.continuous_input_enabled:
            error_bound = (self.max_input - self.min_input) / 2.0

            goal_min_distance = self._input_modulus(
                self.goal.position - measurement, -error_bound, error_bound)
            setpoint_min_distance = self._input_modulus(
                self.setpoint.position - measurement, -error_bound, error_bound)

            self.goal.position = goal_min_distance + measurement
            self.setpoint.position = setpoint_min_distance + measurement

        # Generate next setpoint using trapezoidal profile
        self.setpoint = self.profile.calculate(
            self.controller.period, self.setpoint, self.goal)

        # Calculate PID output
        return self.controller.calculate(measurement, self.setpoint.position)

    def at_goal(self) -> bool:
        """Check if we've reached the goal."""
        return self.at_setpoint() and self.goal == self.setpoint

    def at_setpoint(self) -> bool:
        """Check if we've reached the current setpoint."""
        return self.controller.at_setpoint()

    def reset(self, measurement=None):
        """Reset the controller."""
        self.controller.reset()
        if measurement is not None:
            if isinstance(measurement, (int, float)):
                self.setpoint = TrapezoidProfileState(float(measurement), 0.0)
            elif isinstance(measurement, TrapezoidProfileState):
                self.setpoint = measurement

    def get_goal(self) -> TrapezoidProfileState:
        """Get the current goal."""
        return self.goal

    def get_setpoint(self) -> TrapezoidProfileState:
        """Get the current setpoint."""
        return self.setpoint

    def get_position_error(self) -> float:
        """Get position error."""
        return self.controller.get_error()

    @staticmethod
    def _input_modulus(input_val: float, min_input: float, max_input: float) -> float:
        """Calculate input modulus for continuous input."""
        modulus = max_input - min_input
        num_max = int((input_val - min_input) / modulus)
        return input_val - num_max * modulus


# Example usage
if __name__ == "__main__":
    # Basic PID Controller example
    print("=== Basic PID Controller Example ===")
    pid = PIDController(kp=1.0, ki=0.1, kd=0.05)
    pid.set_setpoint(10.0)
    pid.set_tolerance(0.1)

    measurement = 0.0
    for i in range(50):
        output = pid.calculate(measurement)
        # Simulate a simple system (integrator)
        measurement += output * 0.02  # Simple integration
        print(f"Step {i + 1}: measurement={measurement:.3f}, output={output:.3f}, at_setpoint={pid.at_setpoint()}")

        if pid.at_setpoint():
            print("Reached setpoint!")
            break

    print("\n=== Profiled PID Controller Example ===")
    # Profiled PID Controller example
    constraints = TrapezoidProfileConstraints(max_velocity=5.0, max_acceleration=10.0)
    profiled_pid = ProfiledPIDController(kp=2.0, ki=0.0, kd=0.1, constraints=constraints)
    profiled_pid.set_goal(20.0)
    profiled_pid.set_tolerance(0.1, 0.5)

    measurement = 0.0
    velocity = 0.0

    for i in range(100):
        output = profiled_pid.calculate(measurement)

        # Simulate system dynamics
        acceleration = output
        velocity += acceleration * 0.02
        measurement += velocity * 0.02

        setpoint = profiled_pid.get_setpoint()
        goal = profiled_pid.get_goal()

        if i % 10 == 0:  # Print every 10th step
            print(f"Step {i + 1}: pos={measurement:.2f}, vel={velocity:.2f}, "
                  f"setpoint_pos={setpoint.position:.2f}, setpoint_vel={setpoint.velocity:.2f}, "
                  f"output={output:.2f}")

        if profiled_pid.at_goal():
            print(f"Reached goal at step {i + 1}!")
            break
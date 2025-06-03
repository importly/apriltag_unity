import time
import math
from typing import Optional, Tuple
from dataclasses import dataclass

from PIDController import PIDController

@dataclass
class TrapezoidProfileState:
    """State for trapezoidal motion profile (position, velocity)."""
    position: float = 0.0
    velocity: float = 0.0

    def __eq__(self, other) -> bool:
        if not isinstance(other, TrapezoidProfileState):
            return False
        return (abs(self.position - other.position) < 1e-9 and
                abs(self.velocity - other.velocity) < 1e-9)


@dataclass
class TrapezoidProfileConstraints:
    """Constraints for trapezoidal motion profile."""
    max_velocity: float
    max_acceleration: float


class TrapezoidProfile:
    """
    Trapezoidal motion profile generator.
    """

    def __init__(self, constraints: TrapezoidProfileConstraints):
        self.constraints = constraints

    def calculate(self, dt: float, current: TrapezoidProfileState,
                  goal: TrapezoidProfileState) -> TrapezoidProfileState:
        """
        Calculate the next state in the trapezoidal profile.

        Args:
            dt: Time step
            current: Current state
            goal: Goal state

        Returns:
            Next state
        """
        direction = 1 if goal.position >= current.position else -1
        max_vel = self.constraints.max_velocity * direction
        max_accel = self.constraints.max_acceleration

        # Distance to goal
        distance_to_goal = abs(goal.position - current.position)

        # If we're close enough, just go to goal
        if distance_to_goal < 1e-6:
            return TrapezoidProfileState(goal.position, 0.0)

        # Calculate what velocity we need to decelerate to goal
        # v² = u² + 2as, solving for v when we want to end at goal velocity
        decel_vel_sq = goal.velocity * goal.velocity + 2 * max_accel * distance_to_goal
        decel_vel = math.sqrt(max(0, decel_vel_sq)) * direction

        # Choose velocity: either max velocity or deceleration velocity
        if abs(decel_vel) < abs(max_vel):
            target_vel = decel_vel
        else:
            target_vel = max_vel

        # Calculate acceleration needed
        vel_error = target_vel - current.velocity
        if abs(vel_error) < max_accel * dt:
            # Can reach target velocity in this timestep
            next_velocity = target_vel
        else:
            # Accelerate toward target velocity
            accel = max_accel if vel_error > 0 else -max_accel
            next_velocity = current.velocity + accel * dt

        # Calculate next position
        avg_velocity = (current.velocity + next_velocity) / 2
        next_position = current.position + avg_velocity * dt

        return TrapezoidProfileState(next_position, next_velocity)


from __future__ import annotations
import math
from typing import Optional

from subsystems.navigation import RobotConfig, RobotNavigationSystem
from .scheduler import Subsystem


class DriveSubsystem(Subsystem):
    """Subsystem wrapping :class:`RobotNavigationSystem` for the scheduler."""

    def __init__(self, config: Optional[RobotConfig] = None) -> None:
        super().__init__()
        self.config = config or RobotConfig()
        self.nav = RobotNavigationSystem(self.config)
        self.nav.set_motor_command_callback(self._send_motor_commands)

    # ------------------------------------------------------------------ util
    def _send_motor_commands(self, left: float, right: float) -> None:
        """Placeholder for sending commands to the drive motors."""
        print(f"[Drive] Left={left:.2f} Right={right:.2f}")

    # -------------------------------------------------------------- subsystem
    def periodic(self) -> None:  # type: ignore[override]
        status = self.nav.get_status()
        if status.get("enabled"):
            print(
                f"[Drive] dist={status['distance_to_target']:.2f}m, "
                f"angle={status['angle_error_deg']:.1f}°"
            )

    def close(self) -> None:  # type: ignore[override]
        self.nav.stop()
        self.nav.stop_control_loop()

    # -------------------------------------------------------------- commands
    def update_pose(self, x: float, y: float, yaw: float) -> None:
        self.nav.update_pose_from_vision(x, y, yaw)

    def navigate_to(self, x: float, y: float, yaw_deg: float) -> None:
        yaw_rad = math.radians(yaw_deg)
        self.nav.navigate_to(x, y, yaw_rad)

    def stop(self) -> None:
        self.nav.stop()

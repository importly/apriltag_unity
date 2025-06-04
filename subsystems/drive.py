from __future__ import annotations
import math
import socket
import time
from typing import Optional

from subsystems.navigation import RobotConfig, RobotNavigationSystem
from .scheduler import Subsystem
from util.logging_utils import get_robot_logger

logger = get_robot_logger(__name__)


class DriveSubsystem(Subsystem):
    """Subsystem wrapping :class:`RobotNavigationSystem` for the scheduler."""

    def __init__(self, config: Optional[RobotConfig] = None,
                 spu_host: str = "localhost", spu_port: int = 5008) -> None:
        super().__init__()
        self.config = config or RobotConfig()
        self.nav = RobotNavigationSystem(self.config)
        self.nav.set_motor_command_callback(self._send_motor_commands)
        self.spu_host = spu_host
        self.spu_port = spu_port
        self.sock = self._connect_to_spu(spu_host, spu_port)

    # ------------------------------------------------------------------ util
    def _connect_to_spu(self, host: str, port: int, retry_delay: float = 1.0):
        """Keep trying to connect to the SPU until successful."""
        while True:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                logger.info("[TCP] Connecting to SPU on %s:%s…", host, port)
                sock.connect((host, port))
                logger.info("[TCP] ▶︎ Connected to SPU")
                return sock
            except Exception as e:
                logger.error(
                    "[TCP] ERROR connecting to SPU: %s. Retrying in %ss…",
                    e,
                    retry_delay,
                )
                time.sleep(retry_delay)

    def _send_motor_commands(self, left: float, right: float) -> None:
        """Send motor commands to the SPU over TCP."""
        try:
            self.sock.sendall(f"{left:.2f},{right:.2f}".strip().encode("utf8"))
            logger.debug("Sent motor commands: %s | Current Position: %.2f, %.2f, %.2f",
                         f"{left:.2f},{right:.2f}".strip(), self.nav.controller.current_pose.x,
                         self.nav.controller.current_pose.y, self.nav.controller.current_pose.yaw)

        except Exception as e:
            logger.error("[TCP] ERROR sending to SPU: %s. Reconnecting…", e)
            self.sock.close()
            self.sock = self._connect_to_spu(self.spu_host, self.spu_port)

    # -------------------------------------------------------------- subsystem
    def periodic(self) -> None:  # type: ignore[override]
        status = self.nav.get_status()
        if status.get("enabled"):
            logger.info(
                "[Drive] dist=%.2fm, angle=%.1f°",
                status["distance_to_target"],
                status["angle_error_deg"],
            )

    def close(self) -> None:  # type: ignore[override]
        self.nav.stop()
        self.nav.stop_control_loop()
        if self.sock:
            self.sock.close()

    # -------------------------------------------------------------- commands
    def update_pose(self, x: float, y: float, yaw: float) -> None:
        self.nav.update_pose_from_vision(x, y, yaw)

    def navigate_to(self, x: float, y: float, yaw_deg: float) -> None:
        yaw_rad = math.radians(yaw_deg)
        self.nav.navigate_to(x, y, yaw_rad)

    def stop(self) -> None:
        self.nav.stop()

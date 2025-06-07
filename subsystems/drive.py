from __future__ import annotations
import math
import time
from typing import Optional

import serial

from subsystems.navigation import RobotConfig, RobotNavigationSystem
from .scheduler import Subsystem
from util.logging_utils import get_robot_logger

logger = get_robot_logger(__name__)


class DriveSubsystem(Subsystem):
    """Subsystem wrapping :class:`RobotNavigationSystem` for the scheduler."""

    def __init__(
        self,
        config: Optional[RobotConfig] = None,
        serial_port: str = "COM7",
        baud_rate: int = 9600,
    ) -> None:
        super().__init__()
        self.config = config or RobotConfig()
        self.nav = RobotNavigationSystem(self.config)
        self.nav.set_motor_command_callback(self._send_motor_commands)
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.ser = self._connect_to_serial(serial_port, baud_rate)

    # ------------------------------------------------------------------ util
    def _connect_to_serial(
        self, port: str, baud_rate: int, retry_delay: float = 1.0
    ):
        """Keep trying to open the serial port until successful."""
        while True:
            try:
                ser = serial.Serial(port=port, baudrate=baud_rate, timeout=1)
                logger.info(
                    "[Serial] ->︎ Connected to %s @ %sbps",
                    port,
                    baud_rate,
                )
                return ser
            except Exception as e:
                logger.error(
                    "[Serial] ERROR opening %s: %s. Retrying in %ss…",
                    port,
                    e,
                    retry_delay,
                )
                time.sleep(retry_delay)

    def _encode_drive_packet(self, channel: int, speed: float) -> bytes:
        raw = int(((speed + 1.0) / 2.0) * 255)
        raw = max(0, min(255, raw))
        lsb = raw & 0x7F
        msb = (raw >> 7) & 0x7F
        return bytes([0x84, channel, lsb, msb])

    def _send_motor_commands(self, left: float, right: float) -> None:
        """Send motor commands to the robot over Serial."""
        try:
            self.ser.write(self._encode_drive_packet(0, left))
            self.ser.write(self._encode_drive_packet(1, right))
            logger.debug(
                "Sent motor commands L:%.2f R:%.2f | Pose: %.2f, %.2f, %.2f",
                left,
                right,
                self.nav.controller.current_pose.x,
                self.nav.controller.current_pose.y,
                self.nav.controller.current_pose.yaw,
            )

        except Exception as e:
            logger.error("[Serial] ERROR sending commands: %s. Reconnecting…", e)
            try:
                self.ser.close()
            finally:
                self.ser = self._connect_to_serial(self.serial_port, self.baud_rate)

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
        if self.ser:
            self.ser.close()

    # -------------------------------------------------------------- commands
    def update_pose(self, x: float, y: float, yaw: float) -> None:
        self.nav.update_pose_from_vision(x, y, yaw)

    def navigate_to(self, x: float, y: float, yaw_deg: float) -> None:
        yaw_rad = math.radians(yaw_deg)
        self.nav.navigate_to(x, y, yaw_rad)

    def stop(self) -> None:
        self.nav.stop()

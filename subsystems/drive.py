from __future__ import annotations
import math
import socket
import time
from typing import Optional

from subsystems.navigation import RobotConfig, RobotNavigationSystem
from .scheduler import Subsystem


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
                print(f"[TCP] Connecting to SPU on {host}:{port}…")
                sock.connect((host, port))
                print("[TCP] ▶︎ Connected to SPU")
                return sock
            except Exception as e:
                print(
                    f"[TCP] ERROR connecting to SPU: {e}. Retrying in {retry_delay}s…"
                )
                time.sleep(retry_delay)

    def _send_motor_commands(self, left: float, right: float) -> None:
        """Send motor commands to the SPU over TCP."""
        line = f"{left:.2f},{right:.2f}\r\n"
        try:
            self.sock.sendall(line.encode("utf8"))
        except Exception as e:
            print(f"[TCP] ERROR sending to SPU: {e}. Reconnecting…")
            self.sock.close()
            self.sock = self._connect_to_spu(self.spu_host, self.spu_port)

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

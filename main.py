"""Simple CLI interface for commanding the robot."""

from subsystems.scheduler import CommandScheduler
from subsystems.drive import DriveSubsystem
from subsystems.vision import VisionSubsystem
from util.logging_utils import get_robot_logger

logger = get_robot_logger(__name__)


def main() -> None:

    scheduler = CommandScheduler()
    drive = DriveSubsystem()
    scheduler.register(drive)
    vision = VisionSubsystem()
    scheduler.register(vision)

    # Initialize robot at origin facing north
    drive.update_pose(0.0, 0.0, 0.0)

    try:
        while True:
            scheduler.run()
            # cmd = input(
            #     "Enter target x y heading (deg) or 'q' to quit: "
            # ).strip()
            # if cmd.lower() in {"q", "quit", "exit"}:
            #     break
            # if not cmd:
            #     continue
            # try:
            #     x_str, y_str, h_str = cmd.split()
            #     x = float(x_str)
            #     y = float(y_str)
            #     heading = float(h_str)
            # except ValueError:
            #     print("Please enter three numeric values.")
            #     continue
            #
            # drive.navigate_to(x, y, heading)

    finally:
        scheduler.shutdown()
        logger.info("Robot program exiting")


if __name__ == "__main__":
    main()

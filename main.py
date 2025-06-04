import cv2
from subsystems.scheduler import CommandScheduler
from subsystems.drive import DriveSubsystem
from subsystems.vision import VisionSubsystem
from util.logging_utils import get_robot_logger

logger = get_robot_logger(__name__)


def main() -> None:
    logger.info("Robot program starting")
    scheduler = CommandScheduler()
    drive = DriveSubsystem()
    scheduler.register(drive)
    # vision = VisionSubsystem()
    # scheduler.register(vision)

    try:
        while True:
            scheduler.run()
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    finally:
        scheduler.shutdown()
        logger.info("Robot program exiting")


if __name__ == "__main__":
    main()

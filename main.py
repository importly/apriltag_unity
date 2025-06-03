#!/usr/bin/env python3
"""Entry point for the robot program using a simple command-based structure."""

import cv2
from subsystems.scheduler import CommandScheduler
from subsystems.vision import VisionSubsystem


def main() -> None:
    scheduler = CommandScheduler()
    vision = VisionSubsystem()
    scheduler.register(vision)

    try:
        while True:
            scheduler.run()
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    finally:
        scheduler.shutdown()


if __name__ == "__main__":
    main()

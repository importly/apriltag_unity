from subsystems.scheduler import CommandScheduler
from subsystems.vision import VisionSubsystem
from util.logging_utils import get_robot_logger
from subsystems.geometry import Pose3D, Translation3D, Rotation3D

logger = get_robot_logger(__name__)

# the program looks like this is because it is designed to be run on a robot, so it has a main function that runs the scheduler
# but right now it is just a simple program that runs the vision subsystem
def main() -> None:
    scheduler = CommandScheduler()
    camera_poses = [ # camera relative to center of robot, these are the poses of the cameras in the robot's coordinate system
        Pose3D(Translation3D(0.1, 0.0, 0.2), Rotation3D(0.0, 0.0, 0.0)),
        Pose3D(Translation3D(-0.1, 0.0, 0.2), Rotation3D(0.0, 0.0, 0.0)),
    ]
    field_apriltag_poses = { # global poses of the apriltags in the field coordinate system
        1: Pose3D(Translation3D(1.0, 2.0, 0.0), Rotation3D(0.0, 0.0, 0.0)),
        2: Pose3D(Translation3D(2.0, 3.0, 0.0), Rotation3D(0.0, 0.0, 0.0)),
        3: Pose3D(Translation3D(3.0, 4.0, 0.0), Rotation3D(0.0, 0.0, 0.0)),
    }
    vision = VisionSubsystem( # all possible parameters are listed here, you can change them to suit your needs
        camera_indices=[1, 2],
        tag_size=0.165,
        frame_width=1600,
        frame_height=1200,
        frame_rate=50,
        spu_host="localhost",
        spu_port=5008,
        threads=8,
        families="tag36h11",
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0, # idk what this does, but it is a parameter that you can change
        camera_poses=camera_poses,
        field_apriltag_poses=field_apriltag_poses,
    )
    scheduler.register(vision) # currently configured for just vision, soloman can expand to driving and stuff as needed later.


    try:
        while True:
            scheduler.run()

    finally:
        scheduler.shutdown()
        logger.info("Robot program exiting")


if __name__ == "__main__":
    main()

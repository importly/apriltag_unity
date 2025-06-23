# AprilTag Unity Project

This repository bundles Python utilities that work with Unity's RoboCor tools to provide AprilTag-based navigation and drive control.  Fused robot poses are streamed to the RoboCor SPU (socket processing unit) and then consumed inside Unity for autonomous docking and path following.

## Features and Capabilities

- **Vision subsystem**
  - Multiple calibrated cameras are opened with OpenCV. Calibration files are stored as `.npy` or `.yaml` in the project root.
  - AprilTags are detected using the high performance `pupil_apriltags` detector.
  - For each detection the 3D pose of the tag relative to the camera is solved.  These measurements are transformed into robot space and fused across cameras.
  - The resulting robot pose is sent over a persistent TCP socket to the SPU.  RoboCor's utilities inside Unity read this stream to drive the robot and visualise its position.
  - If no valid tags are found the subsystem logs the absence so Unity can fall back to odometry.

- **Navigation and Drive system**
  - `RobotNavigationSystem` implements full differential-drive kinematics.  Wheel speeds and body twists are converted using rotation matrices and geometry helpers.
  - `DifferentialDriveController` uses profiled PID loops with trapezoidal velocity constraints.  This enables smooth acceleration and deceleration while respecting physical limits.
  - Wheel commands are integrated to update odometry.  The system continuously normalises angles and applies deadband logic to handle wraparound.
  - `DriveSubsystem` packages the controller for the scheduler and encodes commands for the Arduino motor driver over serial.
  - These calculations – wheel speed conversions, odometry integration and angle normalisation – allow precise navigation to waypoints or docking targets.

- **Pose estimation**
  - The `PoseEstimator` maintains a history of odometry samples and vision updates.  Measurements are weighted and interpolated over time to provide a stable pose estimate.
  - This fused pose is what gets streamed to the SPU for use by RoboCor in Unity.

- **Scheduler**
  - A tiny `CommandScheduler` repeatedly calls each subsystem’s `periodic` method so the program behaves like a typical robot control loop.

- **RealSense heading tracker**
  - `realsense_heading.py` runs a background thread that integrates RealSense gyro data.  It removes bias through calibration and uses a short median filter to reject outliers.
  - An optional Tkinter GUI draws a compass, displays confidence, and shows total rotations.

- **Testing utilities**
  - `test.py` stress tests the motor interface and confirms the serial connection to the Arduino.
  - Logging helpers record events to `robot.log` with rotation to avoid huge log files.

- **Environment and calibration**
  - The `environment.yml` file lists all Python packages – OpenCV, pupil-apriltags, pyrealsense2 and others needed for the robot software.
  - Example calibration and pose data are stored in the repository for quick setup.

This documentation maps the delivered features to project payments and gives a complete overview of what each module provides.

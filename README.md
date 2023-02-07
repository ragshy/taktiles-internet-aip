# Audio Internet Project

For image processing part of our projetct, a method for the head orientation and localisation of a listener and talker in a closed room environment is developed. 

## Get started:

Setup for 2 Users:

  - 1 Laptop each
  - 2 cameras each
  - on the same network

0.  Pre-requisites:
    - Python 3.8 or higher
    - Internet connection
    - Python virtual environment (recommended)

1.  Install requirements
    ```
    pip install -r requirements.txt
    ```

    ! Apple Silicon devices: 
    
    [Mediapipe](https://google.github.io/mediapipe/) library (for M1 Chip and so on) may have to be installed separately

2.  Calibrate Cameras
    - For individual camera calibration, run
        ```
        python cam_calibration.py
        ```
      Needs checkerboard images made with the respective camera, see there for more information.
    - For stereo camera calibration, run
        ```
        python stereocam_calibration.py
        ```
      Needs checkerboard images made with the cameras simultaneously (see SimultCapture_StereoSetup.py).

3.  Run one user as server
    ```
    python track.py -s
    ```

    Run other user as client (can also be run stand-alone for debugging)
    ```
    python track.py
    ```
    ! Make sure that track.py has the correct IP-Adress (see line 17) of the server device

## Files and Folders descriptions:
  - ### /Camera_Calibration:

      contains images and scripts to calibrate cameras individually and stereo
  - ### /data:

      contains data in .csv format from Optitrack (used as reference data) and from our project to be able to compare and measure performance
  - ### /Data_Analysis:

      measures performance and plots using /data in MATLAB

  - ### TCPConnection.py

    contains a Server class and Client class for TCP (or UDP or other Network) connection
  
  - ### utils.py

    contains all necessary helper functions for computation of triangulation, perspective-n-point problem, facial landmark detection etc.

  - ### track.py

    main script that can run as server or as client





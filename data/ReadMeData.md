ReadMe about the Datasets
#########################

We recorded 4 scenarios in the Optitrack chamber: 

Sc1: 
    - Head Orientation from close and then from far
    - The order of actions is: pitch, roll, yaw

Sc2: 
    - Side to Side
    - Start from Middle of main camera, then go to the right, and then left

Sc3:
    - Close and Far
    - Stay in middle of camera, step back and then step forward towards camera

Sc4: 
    - Free talking scenario, freely moving 

The data is named accordingly. There are 4 datasets per scenario, as we have a data from Optitrack and our system per speaker.

The raw data is given as well.

Some preprocessing was done via Matlab, so the script cam_data_preprocess.py and time_alignment.py are not including all processing steps.

Besides the scenarios, dummy data was recorded. The test person did movements in a certain order to simplify mapping and understanding of the data. 
Dummy test order:
    - move to right then back to center
    - move backwards and back to center
    - move down, (squat)
    - Head moving: yaw, pitch, roll
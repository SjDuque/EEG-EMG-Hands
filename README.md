# EEG-EMG-Hands
Controlling a robotic hand using OpenBCI Cyton.

## Training a machine learning model to detect when a hand is closed (openbci/openbci_focus_hand.py)
Programmed a machine learning model to infer when my hand is closed using EMG ([muscle reading](https://docs.openbci.com/GettingStarted/Biosensing-Setups/EMGSetup/)).

Initially the model is untrained so there is no response. Pressing 'T' enables training mode, where I calibrate what actions correspond to green or red. I decided to set green to closing and red to opening my hand.

Notice after training that the robotic hand closes and opens with mine.

https://github.com/user-attachments/assets/c013bcee-4476-4542-a547-f4df26d3af68



## Controlling a robotic arm using mediapipe/camera (src/MediapipeEsp32.py)
Using hand landmark detection from Mediapipe, I calculated the angles of my fingers and mapped them to the servos on the robotic hand.



https://github.com/user-attachments/assets/08382008-1f0d-4633-aaa4-c096e12325b7




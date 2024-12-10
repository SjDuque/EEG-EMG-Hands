# EEG-EMG-Hands
Controlling a robotic hand using OpenBCI Cyton.

## Training a machine learning model to detect when a hand is closed (openbci/openbci_focus_hand.py)
Programmed a machine learning model to infer when my hand is closed using [EMG](https://docs.openbci.com/GettingStarted/Biosensing-Setups/EMGSetup/) (aka muscle reading).

Initially the model is untrained so there is no response. Pressing 'T' enables training mode, where I calibrate what actions correspond to green or red. I decided to set green to closing and red to opening my hand.

Notice after training that the robotic hand closes and opens with mine.

Note: There is no webcam use in this example, I am using an OpenBCI Cyton to read muscle activity using electrodes attached to my forearm.

https://github.com/user-attachments/assets/c013bcee-4476-4542-a547-f4df26d3af68

## Controlling a robotic arm using mediapipe/camera (src/MediapipeEsp32.py)
Using hand landmark detection from Mediapipe, I calculated the angles of my fingers and mapped them to the servos on the robotic hand.

https://github.com/user-attachments/assets/08382008-1f0d-4633-aaa4-c096e12325b7

## Performance

`src/mediapipe_lsl.py`
Using a 260 hz Camera
- M4 Mac Mini 16 GB: ~ 230 FPS

## FOR MAC USERS
Make sure to disable 'Reactions' by clicking the green camera icon and ensuring 
Reactions isn't enabled. This is especially important when running at over 200 FPS
as MacOS will randomly detect a thumbs up, freezing the program.

# EEG-EMG-Hands
Controlling a robotic hand using OpenBCI Cyton.

## Training a machine learning model to detect when a hand is closed (openbci/openbci_focus_hand.py)
Programmed a machine learning model to infer when my ahnd is closed using EMG (muscle reading).

Initially the model is untrained so there is no response. Pressing 'T' enables training mode, where I calibrate what actions correspond to green or red. I decided to set green to closing and red to opening my hand.

Notice after training that the robotic hand closes and opens with mine.

![ML Hand Close](https://github.com/user-attachments/assets/b4592ec4-718b-437e-9767-910aab9f0188)

## Controlling a robotic arm using mediapipe/camera (src/MediapipeEsp32.py)
Using hand landmark detection from Mediapipe, I calculated the angles of my fingers and mapped them to the servos on the robotic hand.

![Hand Mediapipe](https://github.com/user-attachments/assets/38943620-b3ae-410f-8470-c48883c280af)


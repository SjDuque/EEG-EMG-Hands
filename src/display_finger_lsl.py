import pylsl
import numpy as np
import cv2
import time

def run_receiver_simple():
    # Resolve streams
    print("Looking for HandLandmarks stream...")
    streams_landmarks = pylsl.resolve_byprop("name", "HandLandmarks", timeout=10)
    if not streams_landmarks:
        print("No HandLandmarks stream found.")
        return

    print("Looking for HandAngles stream...")
    streams_angles = pylsl.resolve_byprop("name", "HandAngles", timeout=10)
    if not streams_angles:
        print("No HandAngles stream found.")
        return

    inlet_landmarks = pylsl.StreamInlet(streams_landmarks[0])
    inlet_angles = pylsl.StreamInlet(streams_angles[0])
    
    # Print the inlet info
    print("Landmarks Inlet Info:")
    print(inlet_landmarks.info().as_xml())
    print("Angles Inlet Info:")
    print(inlet_angles.info().as_xml())

    # Assume 21 landmarks, each with x,y,z
    num_landmarks = 21
    img_size = 600
    scale = img_size  # to scale x,y from [0,1] to image coordinates

    cv2.namedWindow("HandReplay", cv2.WINDOW_NORMAL)

    while True:
        # Pull samples
        landmark_sample = inlet_landmarks.pull_chunk(timeout=0.01)[0]
        angle_sample = inlet_angles.pull_chunk(timeout=0.01)[0]
        
        if landmark_sample and angle_sample:
            landmark_sample = landmark_sample[0]
            angle_sample = angle_sample[0]
            
        # print(landmark_sample)
        # print(angle_sample)

        if landmark_sample is not None and len(landmark_sample) == num_landmarks*3:
            # Convert to (21,3)
            coords = np.array(landmark_sample).reshape(num_landmarks, 3)

            # Draw on a white image
            img = np.ones((img_size, img_size, 3), dtype=np.uint8)*255

            for (x,y,z) in coords:
                if not np.isnan(x):
                    px = int(x * scale)
                    py = int(y * scale)
                    cv2.circle(img, (px, py), 5, (0,0,255), -1)
                print 
            
            # Display angles if available
            angles_text = []
            if angle_sample is not None:
                for a in angle_sample:
                    if np.isnan(a):
                        angles_text.append("NaN")
                        continue
                    a /= np.pi
                    a *= 100
                    angles_text.append(f"{a:.0f}%")
                angles_text = ", ".join(angles_text)
                cv2.putText(img, f"Angles: {angles_text}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

            cv2.imshow("HandReplay", img)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_receiver_simple()

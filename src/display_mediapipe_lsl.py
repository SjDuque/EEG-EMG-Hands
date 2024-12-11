import pylsl
import numpy as np
import cv2

def run_receiver_simple():
    # Resolve streams
    print("Looking for HandLandmarks stream...")
    streams_landmarks = pylsl.resolve_stream("name", "HandLandmarks")
    if not streams_landmarks:
        print("No HandLandmarks stream found.")
        return

    print("Looking for FingerPercentages stream...")
    streams_angles = pylsl.resolve_stream("name", "FingerPercentages")
    if not streams_angles:
        print("No FingerPercentages stream found.")
        return

    inlet_landmarks = pylsl.StreamInlet(streams_landmarks[0], processing_flags=pylsl.proc_ALL)
    inlet_percentages = pylsl.StreamInlet(streams_angles[0], processing_flags=pylsl.proc_ALL)
    
    # Print the inlet info
    print("Landmarks Inlet Info:")
    print(inlet_landmarks.info().as_xml())
    print("Angles Inlet Info:")
    print(inlet_percentages.info().as_xml())

    # Assume 21 landmarks, each with x,y,z
    num_landmarks = 21
    img_size = 600
    scale = img_size  # to scale x,y from [0,1] to image coordinates
    
    # Connections between landmarks
    # (color, landmark_list)
    black = (0,0,0)
    white = (255,255,255)
    green = (0,255,0)
    red = (0,0,255)
    blue = (255,0,0)
    yellow = (0,255,255)
    orange = (0,165,255)
    violet = (255,0,255)
    connections = {
        "palm":     (orange,  [0, 1, 5, 9, 13, 17, 0]),
        "thumb":    (green,  [1, 2, 3, 4]),
        "index":    (red,    [5, 6, 7, 8]),
        "middle":   (blue,   [9, 10, 11, 12]),
        "ring":     (yellow, [13, 14, 15, 16]),
        "pinky":    (violet, [17, 18, 19, 20]),
    }
    
    # Thumb, index, middle, ring, pinky
    finger_thresholds = [(0.8, 0.9), (0.7, 0.875), (0.75, 0.875), (0.7, 0.8), (0.7, 0.825)]
    # Ring finger and pinky are often linked to another finger, so their thresholds
    # may be lower than the others.
    # For example, when I move my left pinky, my ring finger moves.
    # Or if I move my right middle finger, my ring finger moves.

    cv2.namedWindow("HandReplay", cv2.WINDOW_NORMAL)

    while True:
        # Pull samples
        landmark_sample = inlet_landmarks.pull_chunk(timeout=0.01)[0]
        percentage_sample = inlet_percentages.pull_chunk(timeout=0.01)[0]
        
        if landmark_sample and percentage_sample:
            landmark_sample = landmark_sample[-1]
            percentage_sample = percentage_sample[-1]

        if landmark_sample is not None and len(landmark_sample) == num_landmarks*3:
            # Convert to (21,3)
            coords = np.array(landmark_sample).reshape(num_landmarks, 3)
            
            # Draw on a blank image
            img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            
            if not np.isnan(coords).any():
                
                # Draw connections
                for color, landmark_list in connections.values():
                    thickness = 5
                    for i in range(len(landmark_list)-1):
                        cv2.line(img, tuple((coords[landmark_list[i]][:2]*scale).astype(int)), tuple((coords[landmark_list[i+1]][:2]*scale).astype(int)), color, thickness)

                # Display percentages if available
                percentages_text = []
                if percentage_sample is not None:
                    for i, a in enumerate(percentage_sample):
                        if np.isnan(a):
                            percentages_text.append("NaN")
                            continue
                        percentages_text.append(f"{int(a*100)}%")
                    percentages_text = ", ".join(percentages_text)
                    cv2.putText(img, f"Percentages: {percentages_text}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, white, 2)
                
                thresholds_text = []
                if percentage_sample is not None:
                    for i, a in enumerate(percentage_sample):
                        if np.isnan(a):
                            thresholds_text.append("NaN")
                            continue
                        finger_threshold = finger_thresholds[i]
                        a = (a - finger_threshold[0]) / (finger_threshold[1] - finger_threshold[0]) * 100
                        a = np.clip(a, 0, 100)
                        
                        thresholds_text.append(f"{int(a)}%")
                    thresholds_text = ", ".join(thresholds_text)
                    cv2.putText(img, f"Thresholds: {thresholds_text}", (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, white, 2)

            cv2.imshow("HandReplay", img)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_receiver_simple()

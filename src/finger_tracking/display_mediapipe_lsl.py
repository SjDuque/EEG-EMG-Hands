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
    img_size = 360
    scale = img_size  # to scale x,y from [0,1] to image coordinates
    font_scale = 0.45
    font_thickness = 2
    
    # Connections between landmarks
    # (color, landmark_list)
    black = (0,0,0)
    white = (255,255,255)
    green = (0,255,0)
    red = (0,0,255)
    blue = (255,255,0)
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

    while True:
        # Pull samples
        landmark_samples, _ = inlet_landmarks.pull_chunk(timeout=0.01)
        percentage_samples, _ = inlet_percentages.pull_chunk(timeout=0.01)
        
        if landmark_samples and percentage_samples:
            landmark_sample = landmark_samples[-1]
            percentage_sample = percentage_samples[-1]
        else:
            landmark_sample = None
            percentage_sample = None

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
                        pt1 = tuple((coords[landmark_list[i]][:2]*scale).astype(int))
                        pt2 = tuple((coords[landmark_list[i+1]][:2]*scale).astype(int))
                        cv2.line(img, pt1, pt2, color, thickness)

                # Display percentages if available
                if percentage_sample is not None:
                    absolute_texts = []
                    relative_texts = []
                    for i, a in enumerate(percentage_sample):
                        if np.isnan(a):
                            absolute_texts.append("NaN")
                            relative_texts.append("NaN")
                            continue
                        absolute_texts.append(f"{int(a*100):3d}%")  # Fixed width with spaces
                        finger_threshold = finger_thresholds[i]
                        relative_a = (a - finger_threshold[0]) / (finger_threshold[1] - finger_threshold[0]) * 100
                        relative_a = np.clip(relative_a, 0, 100)
                        relative_texts.append(f"{int(relative_a):3d}%")  # Fixed width with spaces

                    absolute_text = ", ".join(absolute_texts)
                    relative_text = ", ".join(relative_texts)

                    cv2.putText(img, f"Absolute: {absolute_text}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, white, font_thickness)
                    cv2.putText(img, f"Relative: {relative_text}", (10,55), cv2.FONT_HERSHEY_SIMPLEX, font_scale, white, font_thickness)

            cv2.imshow("Hand Tracking Client", img)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_receiver_simple()

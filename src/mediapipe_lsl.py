import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import warnings
import queue
import pylsl
from itertools import combinations

# Suppress TensorFlow warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# ================================
# Configuration Parameters
# ================================
flip_camera = True
camera_resolution = (640, 360)
mediapipe_resolution_length = 224
display_resolution_length = 360
target_camera_fps = 260
target_mp_fps = 250
target_display_fps = 30

frame_time_display = 1.0 / target_display_fps
frame_time_process = 1.0 / target_mp_fps
stop_event = threading.Event()
status_index = 0
last_status_switch_time = time.perf_counter()

display_frame = None
display_q = queue.Queue(maxsize=2)
process_q = queue.Queue(maxsize=2)
landmark_q = queue.Queue(maxsize=2)
display_landmark_q = queue.Queue(maxsize=2)

left_hand = True
status_labels = ['thumb', 'index', 'middle', 'ring', 'pinky']
status_labels_idx = {label: i for i, label in enumerate(status_labels)}
status_groups = ['thumb', 'index', 'middle', ['ring', 'pinky']]

# Status guides the user on which fingers to raise given a visual prompt
if left_hand:
    status_labels = status_labels[::-1]
    status_labels_idx = {label: i for i, label in enumerate(status_labels)}
    status_groups = status_groups[::-1]

def generate_status_lists():
    result = []
    num_groups = len(status_groups)

    for num_true in range(num_groups + 1):
        for combo in combinations(range(num_groups), num_true):
            status = [False] * len(status_labels)
            for group_index in combo:
                group = status_groups[group_index]
                if isinstance(group, list):
                    for label in group:
                        status[status_labels_idx[label]] = True
                else:
                    status[status_labels_idx[group]] = True
            result.append(status)
    return result

status_lists = generate_status_lists()
status_switch_interval = 5  # seconds

current_status = status_lists[0] if status_lists else [False] * 5

def draw_status_circles(image, status, status_labels):
    radius = 20
    num_circles = len(status)
    image_width = image.shape[1]
    margin = 50
    if num_circles > 1:
        total_spacing = image_width - 2 * margin - (2 * radius * num_circles)
        spacing = total_spacing / (num_circles - 1)
    else:
        spacing = 0
    center_y = 50
    label_offset = 25
    red = (0, 0, 255)
    green = (0, 255, 0)
    black = (0, 0, 0)

    for i in range(num_circles):
        center_x = int(margin + i * (2 * radius + spacing))
        color = green if status[i] else red
        cv2.circle(image, (center_x, center_y), radius, color, -1)

        label = status_labels[i]
        font_scale = 0.5
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        text_x = center_x - text_size[0] // 2
        text_y = center_y + radius + label_offset
        cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)

        text = 'up' if status[i] else 'down'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = center_x - text_size[0] // 2
        text_y = center_y + text_size[1] // 2
        cv2.putText(image, text, (text_x, text_y), font, font_scale, black, thickness)

def capture_camera_frames(stop_event, target_fps, resolution):
    global process_q, display_q

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        stop_event.set()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap.set(cv2.CAP_PROP_FPS, target_fps)

    frame_counter = 0
    last_time = time.perf_counter()
    last_display_update_time = last_time
    last_process_update_time = last_time
    process_time_lost = 0
    display_time_lost = 0

    height, width = resolution[1], resolution[0]
    side = min(height, width)
    x_start = (width - side) // 2
    y_start = (height - side) // 2

    while not stop_event.is_set():
        frame_timestamp = time.perf_counter()
        frame_timestamp_lsl = pylsl.local_clock()
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        if flip_camera:
            frame = cv2.flip(frame, 1)

        frame = frame[y_start:y_start + side, x_start:x_start + side]
        
        delta_process_time = frame_timestamp - last_process_update_time + process_time_lost
        if delta_process_time >= frame_time_process:
            process_frame = cv2.resize(frame, (mediapipe_resolution_length, mediapipe_resolution_length))
            timestamp_frame = (frame_timestamp_lsl, process_frame)
            try:
                process_q.put_nowait(timestamp_frame)
            except queue.Full:
                try:
                    _ = process_q.get_nowait()
                    process_q.put_nowait(timestamp_frame)
                except queue.Empty:
                    pass
            last_process_update_time = frame_timestamp
            process_time_lost = delta_process_time - frame_time_process
            
        delta_display_time = frame_timestamp - last_display_update_time + display_time_lost
        if delta_display_time >= frame_time_display:
            display_frame_resized = cv2.resize(frame, (display_resolution_length, display_resolution_length))
            try:
                display_q.put_nowait(display_frame_resized)
            except queue.Full:
                try:
                    _ = display_q.get_nowait()
                    display_q.put_nowait(display_frame_resized)
                except queue.Empty:
                    pass
            last_display_update_time = frame_timestamp
            display_time_lost = delta_display_time - frame_time_display

        frame_counter += 1
        if frame_timestamp - last_time >= 1.0:
            fps = int(round(frame_counter / (frame_timestamp - last_time)))
            print(f"Camera Capture Rate: {fps} fps")
            frame_counter = 0
            last_time = frame_timestamp

    cap.release()

def process_frames(stop_event):
    global process_q, display_landmark_q

    mp_hands = mp.solutions.hands
    frame_counter = 0
    last_time = time.perf_counter()

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        model_complexity=0
    ) as hands:

        while not stop_event.is_set():
            try:
                # Wait for a new frame with a timeout to allow graceful exit
                lsl_timestamp, frame_to_process = process_q.get(timeout=frame_time_display*2)
            except queue.Empty:
                continue  # Check stop_event again

            if frame_to_process is not None:
                image_rgb = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                else:
                    hand_landmarks = None
                
                # Convert hand landmarks to a flat list
                if hand_landmarks:
                    hand_landmarks_list = [float(coord) for landmark in hand_landmarks.landmark for coord in (landmark.x, landmark.y, landmark.z)]
                else:
                    hand_landmarks_list = []
                    
                # Put the landmark data into the landmark_q
                timestamp_landmarks = (lsl_timestamp, hand_landmarks_list)
                try:
                    landmark_q.put_nowait(timestamp_landmarks)
                except queue.Full:
                    try:
                        _ = landmark_q.get_nowait()
                        landmark_q.put_nowait(timestamp_landmarks)
                    except queue.Empty:
                        pass    
                    
                # Put the landmark data into the display_landmark_q
                try:
                    display_landmark_q.put_nowait(hand_landmarks)
                except queue.Full:
                    try:
                        _ = display_landmark_q.get_nowait()
                        display_landmark_q.put_nowait(hand_landmarks)
                    except queue.Empty:
                        pass
                
                # Calculate the processing rate
                frame_counter += 1
                current_fps_time = time.perf_counter()
                time_diff = current_fps_time - last_time
                if time_diff >= 1.0:
                    fps = int(round(frame_counter / time_diff))
                    print(f"MediaPipe Processing Rate: {fps} fps")
                    frame_counter = 0
                    last_time = current_fps_time        

def calculate_angle_from_points(a, b, c):
    """Calculate the angle between three points a, b, and c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arccos(np.dot(a-b, c-b) / (np.linalg.norm(a-b) * np.linalg.norm(c-b)))
    return radians

def calculate_finger_percentage(landmark_list, joint_set):
    a = np.array(landmark_list[joint_set[0]*3:joint_set[0]*3+3])
    b = np.array(landmark_list[joint_set[1]*3:joint_set[1]*3+3])
    c = np.array(landmark_list[joint_set[2]*3:joint_set[2]*3+3])
    # Calculate the angle between two vectors
    return calculate_angle_from_points(a, b, c) / np.pi

def calculate_finger_percentages(landmark_list, joint_set_list):
    finger_percentages_list = []
    for joint_set in joint_set_list:
        avg_percentage = 0
        for i in range(0, len(joint_set)-2):
            avg_percentage += calculate_finger_percentage(landmark_list, joint_set[i:i+3])
        finger_percentages_list.append(avg_percentage / (len(joint_set)-2))
    
    return finger_percentages_list

def lsl_mp_stream(stop_event):
    global landmark_q

    # Initialize MediaPipe Hands to get landmark names
    mp_hands = mp.solutions.hands
    landmark_names = [item.name for item in mp_hands.HandLandmark]

    # Each landmark has x, y, z coordinates
    channel_names = [f"{name}_{coord}" for name in landmark_names for coord in ['x', 'y', 'z']]
    
    # Define joint sets for each finger's angle
    joint_set_labels = ['thumb', 'index', 'middle', 'ring', 'pinky']
    joint_set_list =   [(1, 2, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]

    # Create LSL StreamInfo 
    landmark_info = pylsl.StreamInfo('HandLandmarks', 'Markers', len(channel_names), target_mp_fps, 'float32', 'HandLandmarks')
    angle_info = pylsl.StreamInfo('FingerPercentages', 'Markers', len(joint_set_labels), target_mp_fps, 'float32', 'FingerPercentages')

    # Add channel labels to the stream's description
    channels = landmark_info.desc().append_child("channels")
    for name in channel_names:
        channels.append_child("channel").append_child_value("label", name)
        
    # Add angle labels to the stream's description
    angles = angle_info.desc().append_child("angles")
    for name in joint_set_labels:
        angles.append_child("angle").append_child_value("label", name)

    # Prepare a NaN sample
    nan_sample = [float('nan')] * len(channel_names)
    
    # Wait for other threads to start
    time.sleep(1)

    # Create the LSL outlets
    landmark_outlet = pylsl.StreamOutlet(landmark_info)
    angle_outlet = pylsl.StreamOutlet(angle_info)

    next_send_time = time.perf_counter()

    frame_counter = 0
    last_time = time.perf_counter()
    
    last_hand_landmarks = nan_sample
    last_timestamp = pylsl.local_clock()

    while not stop_event.is_set():
        current_time = time.perf_counter()

        # If we're behind schedule, adjust without skipping sends
        if current_time >= next_send_time:
            try:
                # Attempt to get the latest landmark data without blocking
                timestamp, hand_landmarks = landmark_q.get_nowait()
            except queue.Empty:
                timestamp = last_timestamp + frame_time_process
                hand_landmarks = last_hand_landmarks

            if not hand_landmarks or len(hand_landmarks) != len(channel_names):
                hand_landmarks = nan_sample
            last_timestamp = timestamp
            last_hand_landmarks = hand_landmarks
    
            calc_angles = calculate_finger_percentages(hand_landmarks, joint_set_list)
            
            # Send the samples
            landmark_outlet.push_sample(hand_landmarks, timestamp)
            angle_outlet.push_sample(calc_angles, timestamp)

            # Update the next send time
            next_send_time += frame_time_process
            # Handle potential drift by resetting the next_send_time
            if current_time > next_send_time:
                # If we're too far behind, reset the next_send_time
                next_send_time = current_time + frame_time_process

            # Calculate and print the stream rate every second
            frame_counter += 1
            if current_time - last_time >= 1.0:
                fps = int(round(frame_counter / (current_time - last_time)))
                print(f"LSL Stream Rate: {fps} fps")
                frame_counter = 0
                last_time = current_time

            # Sleep until the next send time to maintain the target FPS
            sleep_duration = next_send_time - time.perf_counter()
            if sleep_duration > 0:
                time.sleep(sleep_duration)
            else:
                # If sleep_duration is negative, we're behind schedule
                # Continue without sleeping to catch up
                pass

def lsl_status_stream(stop_event):
    global current_status, status_index
    
    # Create LSL StreamInfo for status
    
    status_info = pylsl.StreamInfo('FingerStatus', 'Markers', len(status_labels), 1/status_switch_interval, 'int8', 'FingerStatus')

    # Add channel labels to the stream's description
    channels = status_info.desc().append_child("channels")
    for name in status_labels:
        channels.append_child("channel").append_child_value("label", name)

    # Create the LSL outlet
    status_outlet = pylsl.StreamOutlet(status_info)

    while not stop_event.is_set():
        # Convert boolean status to integers (0 or 1) for LSL
        status_int = [int(status) for status in current_status]
        # Send the status via LSL
        status_outlet.push_sample(status_int)
        # Wait for the switch interval
        time.sleep(status_switch_interval)
        if status_index == 1:
            np.random.shuffle(status_lists)
        # Update the status index
        status_index = (status_index + 1) % len(status_lists)
        current_status = status_lists[status_index]
        print(f"Switched to status list index {status_index}")

def main():
    global stop_event, current_status, status_index, last_status_switch_time

    # Start the camera capture thread
    camera_thread = threading.Thread(
        target=capture_camera_frames,
        args=(stop_event, target_camera_fps, camera_resolution),
        daemon=True
    )
    camera_thread.start()

    # Start the MediaPipe processing thread
    mp_process_thread = threading.Thread(
        target=process_frames,
        args=(stop_event,),
        daemon=True
    )
    mp_process_thread.start()
    
    # Start the LSL landmark stream thread
    lsl_mp_thread = threading.Thread(
        target=lsl_mp_stream,
        args=(stop_event,),
        daemon=True
    )
    lsl_mp_thread.start()

    # Start the LSL status stream thread
    lsl_status_thread = threading.Thread(
        target=lsl_status_stream,
        args=(stop_event,),
        daemon=True
    )
    lsl_status_thread.start()

    # Remove the existing status update logic from the main loop
    # np.random.shuffle(status_lists)  # Already shuffled in status_update_thread

    frame_counter = 0
    last_time = time.perf_counter()

    try:
        while not stop_event.is_set():

            try:
                # Wait for a new frame with a timeout to allow graceful exit
                display_frame = display_q.get(timeout=frame_time_display*2)
            except queue.Empty:
                continue  # Check stop_event again

            try:
                # Wait for a new frame with a timeout to allow graceful exit
                landmarks = display_landmark_q.get(timeout=frame_time_display*2)
            except queue.Empty:
                continue  # Check stop_event again

            if display_frame is not None:
                if landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        display_frame,
                        landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        mp.solutions.drawing_styles.get_default_hand_connections_style()
                    )

            # Draw status circles based on the current_status
            draw_status_circles(display_frame, current_status, status_labels)
            cv2.imshow('MediaPipe Hand Tracking', display_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                stop_event.set()
                break

            current_time = time.perf_counter()
            frame_counter += 1
            if current_time - last_time >= 1.0:
                fps = int(round(frame_counter / (current_time - last_time)))
                print(f"Display Refresh Rate: {fps} fps")
                frame_counter = 0
                last_time = current_time

    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        camera_thread.join()
        mp_process_thread.join()
        lsl_mp_thread.join()
        lsl_status_thread.join()
        cv2.destroyAllWindows()
        print("Program terminated gracefully.")

if __name__ == "__main__":
    main()

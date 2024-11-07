import cv2
import pandas as pd
import numpy as np
import os
import glob
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import mediapipe as mp
import bisect

# =======================
# Data Loading Module
# =======================

def load_video_frames(frames_folder):
    """
    Load and sort video frames based on timestamp extracted from filenames.

    Parameters:
        frames_folder (str): Path to the folder containing camera frames.

    Returns:
        List of tuples: (timestamp, filepath)
    """
    frame_files = glob.glob(os.path.join(frames_folder, 'frame_*.jpg'))
    frames = []
    for file in frame_files:
        filename = os.path.basename(file)
        try:
            timestamp_str = filename[len('frame_'):-len('.jpg')]
            timestamp = float(timestamp_str)
            frames.append((timestamp, file))
        except ValueError:
            print(f"Invalid frame filename (timestamp extraction failed): {filename}")
    frames.sort(key=lambda x: x[0])
    return frames

def load_emg_data(emg_csv):
    """
    Load EMG data from CSV.

    Parameters:
        emg_csv (str): Path to emg.csv file.

    Returns:
        pandas.DataFrame: EMG data sorted by timestamp.
    """
    df = pd.read_csv(emg_csv)
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['timestamp'] = df['timestamp'].astype(float)
    return df

def load_finger_angles(finger_csv):
    """
    Load finger angles data from CSV.

    Parameters:
        finger_csv (str): Path to fingers.csv or finger_angles.csv file.

    Returns:
        pandas.DataFrame: Finger angles data sorted by timestamp.
    """
    df = pd.read_csv(finger_csv)
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['timestamp'] = df['timestamp'].astype(float)
    return df

# =======================
# Visualization Module
# =======================

def create_emg_plots(emg_window_data, channel_names, plot_height=640, plot_width=400):
    """
    Create separate matplotlib plots for each EMG channel without axes and concatenate them vertically.

    Parameters:
        emg_window_data (pd.DataFrame): EMG data within the window.
        channel_names (list): List of EMG channel names to plot.
        plot_height (int): Total height of the plot area in pixels.
        plot_width (int): Width of each EMG plot in pixels.

    Returns:
        numpy.ndarray: Image of the concatenated EMG plots in BGR format.
    """
    num_channels = len(channel_names)
    # Adjust figure size based on plot dimensions
    fig_height = (plot_height / 100) * num_channels / 6  # Adjust height scaling
    fig_width = plot_width / 100  # Width in inches
    fig, axes = plt.subplots(num_channels, 1, figsize=(fig_width, fig_height), dpi=100, constrained_layout=True)

    if num_channels == 1:
        axes = [axes]

    for ax, channel in zip(axes, channel_names):
        ax.plot(emg_window_data['timestamp'], emg_window_data[channel], color='blue', linewidth=1.5)
        # Plot only the latest value as a marker
        if not emg_window_data.empty:
            last_timestamp = emg_window_data['timestamp'].iloc[-1]
            last_value = emg_window_data[channel].iloc[-1]
            ax.scatter(last_timestamp, last_value, color='red', zorder=5, s=20)  # Smaller red dot
        ax.set_yticks([])
        ax.set_xticks([])
        ax.axis('off')  # Remove axes
        ax.set_xlim(emg_window_data['timestamp'].min(), emg_window_data['timestamp'].max())

    # Create a canvas and convert to image
    canvas = FigureCanvas(fig)
    canvas.draw()
    emg_plot_img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    emg_plot_img = emg_plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    emg_plot_img = cv2.cvtColor(emg_plot_img, cv2.COLOR_RGB2BGR)
    plt.close(fig)
    return emg_plot_img


def create_finger_angles_display(finger_angles, finger_names, display_size=(640, 160)):
    """
    Create a separate matplotlib plot to display finger angles as a bar chart.

    Parameters:
        finger_angles (dict): Dictionary with finger names as keys and angles as values.
        finger_names (list): List of finger names to display.
        display_size (tuple): Size of the display (width, height) in pixels.

    Returns:
        numpy.ndarray: Image of the finger angles display in BGR format.
    """
    angles = [finger_angles.get(finger, np.nan) for finger in finger_names]

    fig, ax = plt.subplots(figsize=(display_size[0]/100, display_size[1]/100), dpi=100, constrained_layout=True)

    # Bar chart for finger angles
    bars = ax.bar(finger_names, angles, color='orange')
    ax.set_ylabel('Angle (°)', fontsize=10)
    ax.set_ylim(0, 180)
    ax.set_title('Finger Angles', fontsize=12, pad=15)
    ax.grid(axis='y')

    # Annotate bars with angle values
    for bar, angle in zip(bars, angles):
        height = bar.get_height()
        if not pd.isna(angle):
            ax.text(bar.get_x() + bar.get_width() / 2, height + 2, f"{angle:.1f}°", ha='center', va='bottom', fontsize=10)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, 0, "N/A", ha='center', va='bottom', fontsize=10)

    # Ensure finger names are visible
    ax.set_xticks(range(len(finger_names)))
    ax.set_xticklabels(finger_names, fontsize=10, rotation=0)

    # Remove y-axis labels and ticks for cleaner look
    ax.set_yticks([])

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Create a canvas and convert to image
    canvas = FigureCanvas(fig)
    canvas.draw()
    angles_img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    angles_img = angles_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    angles_img = cv2.cvtColor(angles_img, cv2.COLOR_RGB2BGR)
    plt.close(fig)
    return angles_img

# =======================
# Hand Landmark Module
# =======================

def initialize_hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5):
    """
    Initialize MediaPipe Hands.

    Returns:
        hands: MediaPipe Hands object
        mp_hands: MediaPipe hands module
        mp_drawing: MediaPipe drawing module
    """
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=static_image_mode,
                           max_num_hands=max_num_hands,
                           min_detection_confidence=min_detection_confidence)
    return hands, mp_hands, mp_drawing

def draw_hand_landmarks(image, landmarks, mp_hands, mp_drawing):
    """
    Draw hand landmarks on the image using MediaPipe's drawing utilities.

    Parameters:
        image (numpy.ndarray): The image/frame to draw on.
        landmarks (list): List of hand landmarks.
        mp_hands: MediaPipe hands module.
        mp_drawing: MediaPipe drawing module.

    Returns:
        numpy.ndarray: Image with hand landmarks drawn.
    """
    if landmarks:
        for hand_landmarks in landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))
    return image

# =======================
# Playback Controller
# =======================

class PlaybackController:
    def __init__(self, frames, emg_df, finger_df, emg_window_size_ms=500, playback_speed=1.0, emg_channels=None):
        """
        Initialize the Playback Controller.

        Parameters:
            frames (list): List of tuples (timestamp, filepath)
            emg_df (pd.DataFrame): EMG data
            finger_df (pd.DataFrame): Finger angles data
            emg_window_size_ms (int): EMG window size in milliseconds
            playback_speed (float): Speed multiplier for playback
            emg_channels (list): Specific EMG channels to display
        """
        self.frames = frames
        self.emg_df = emg_df
        self.finger_df = finger_df
        self.emg_window_size_s = emg_window_size_ms / 1000.0
        self.playback_speed = playback_speed
        self.emg_channels = emg_channels if emg_channels else [col for col in emg_df.columns if col.startswith('emg_channel_')]

        # ====================
        # Fix: Specify Only Five Finger Angle Columns
        # ====================
        # Define the expected finger angle column names based on your CSV
        expected_finger_angles = ['THUMB', 'INDEX', 'MIDDLE', 'RING', 'PINKY']
        # Filter the columns to include only the expected finger angles
        self.finger_names = [col for col in expected_finger_angles if col in finger_df.columns]

        if len(self.finger_names) != 5:
            print(f"Warning: Expected 5 finger angle columns, but found {len(self.finger_names)}. Please check the CSV file.")

        # Precompute timestamps for binary search
        self.emg_timestamps = self.emg_df['timestamp'].tolist()
        self.finger_timestamps = self.finger_df['timestamp'].tolist()
        self.frame_timestamps = [frame[0] for frame in self.frames]

    def get_closest_emg_data(self, target_timestamp):
        """
        Retrieve the EMG data closest to the target timestamp using binary search.

        Parameters:
            target_timestamp (float): The target timestamp to match.

        Returns:
            pd.Series: The EMG data row closest to the target timestamp.
        """
        if len(self.emg_timestamps) == 0:
            return pd.Series({col: np.nan for col in self.emg_df.columns})

        idx = bisect.bisect_left(self.emg_timestamps, target_timestamp)
        if idx == 0:
            closest_idx = 0
        elif idx == len(self.emg_timestamps):
            closest_idx = len(self.emg_timestamps) - 1
        else:
            before = self.emg_timestamps[idx - 1]
            after = self.emg_timestamps[idx]
            if abs(after - target_timestamp) < abs(before - target_timestamp):
                closest_idx = idx
            else:
                closest_idx = idx - 1
        return self.emg_df.iloc[closest_idx]

    def get_emg_window(self, target_timestamp):
        """
        Retrieve a window of EMG data up to the target timestamp.

        Parameters:
            target_timestamp (float): The current timestamp.

        Returns:
            pd.DataFrame: EMG data within the specified window (past data only).
        """
        window_start = target_timestamp - self.emg_window_size_s
        window_end = target_timestamp
        window_df = self.emg_df[(self.emg_df['timestamp'] >= window_start) & (self.emg_df['timestamp'] <= window_end)]
        return window_df

    def get_closest_finger_angles(self, target_timestamp):
        """
        Retrieve the finger angles closest to the target timestamp using binary search.

        Parameters:
            target_timestamp (float): The target timestamp to match.

        Returns:
            dict: Finger angles closest to the target timestamp.
        """
        if len(self.finger_timestamps) == 0:
            return {col: np.nan for col in self.finger_names}

        idx = bisect.bisect_left(self.finger_timestamps, target_timestamp)
        if idx == 0:
            closest_idx = 0
        elif idx == len(self.finger_timestamps):
            closest_idx = len(self.finger_timestamps) - 1
        else:
            before = self.finger_timestamps[idx - 1]
            after = self.finger_timestamps[idx]
            if abs(after - target_timestamp) < abs(before - target_timestamp):
                closest_idx = idx
            else:
                closest_idx = idx - 1
        # Return only the finger angle columns
        return self.finger_df.iloc[closest_idx][self.finger_names].to_dict()

    def get_closest_frame_index(self, target_timestamp):
        """
        Get the index of the frame closest to the target timestamp using binary search.

        Parameters:
            target_timestamp (float): The target timestamp to match.

        Returns:
            int: Index of the closest frame.
        """
        if len(self.frame_timestamps) == 0:
            return -1

        idx = bisect.bisect_left(self.frame_timestamps, target_timestamp)
        if idx == 0:
            closest_idx = 0
        elif idx == len(self.frame_timestamps):
            closest_idx = len(self.frame_timestamps) - 1
        else:
            before = self.frame_timestamps[idx - 1]
            after = self.frame_timestamps[idx]
            if abs(after - target_timestamp) < abs(before - target_timestamp):
                closest_idx = idx
            else:
                closest_idx = idx - 1
        return closest_idx

# =======================
# Main Playback Function
# =======================

def main():
    # Configuration Parameters
    data_folder = "EMG Hand Data 20241031_002827"  # Update this path as needed
    emg_window_size_ms = 500
    playback_speed = 1.0  # Adjust as needed (e.g., 1.0 for real-time)
    emg_channels_to_display = None  # Set to an integer or list if specific channels are needed
    display_height = 800  # Total height of the display in pixels, adjustable

    # Validate playback_speed
    if playback_speed <= 0:
        print("Playback speed must be a positive number.")
        return

    # Define paths
    emg_csv = os.path.join(data_folder, 'emg.csv')
    fingers_csv = os.path.join(data_folder, 'finger_angles.csv')  # Ensure this is the correct filename
    frames_folder = os.path.join(data_folder, 'camera_frames')

    # Check if necessary files and folders exist
    if not os.path.exists(emg_csv):
        print(f"EMG data file not found: {emg_csv}")
        return
    if not os.path.exists(fingers_csv):
        print(f"Finger angles data file not found: {fingers_csv}")
        return
    if not os.path.exists(frames_folder):
        print(f"Camera frames folder not found: {frames_folder}")
        return

    # Load data
    print("Loading video frames...")
    frames = load_video_frames(frames_folder)
    if not frames:
        print("No valid video frames found.")
        return
    print(f"Loaded {len(frames)} video frames.")

    print("Loading EMG data...")
    emg_df = load_emg_data(emg_csv)
    if emg_df.empty:
        print("EMG data is empty.")
        return
    print(f"Loaded {len(emg_df)} EMG samples.")

    print("Loading finger angles data...")
    finger_df = load_finger_angles(fingers_csv)
    if finger_df.empty:
        print("Finger angles data is empty.")
        return
    print(f"Loaded {len(finger_df)} finger angles samples.")

    # Determine EMG channels
    emg_channel_cols = [col for col in emg_df.columns if col.startswith('emg_channel_')]
    if emg_channels_to_display is not None:
        if isinstance(emg_channels_to_display, int):
            emg_channel_cols = emg_channel_cols[:emg_channels_to_display]
        elif isinstance(emg_channels_to_display, list):
            emg_channel_cols = [col for col in emg_channels_to_display if col in emg_channel_cols]
    print(f"Displaying EMG channels: {emg_channel_cols}")

    # Initialize Playback Controller
    playback = PlaybackController(
        frames=frames,
        emg_df=emg_df,
        finger_df=finger_df,
        emg_window_size_ms=emg_window_size_ms,
        playback_speed=playback_speed,
        emg_channels=emg_channel_cols
    )

    # Initialize MediaPipe Hands
    hands, mp_hands, mp_drawing = initialize_hands()

    # Create a named window for display
    cv2.namedWindow('Replay', cv2.WINDOW_NORMAL)
    # Calculate display width based on the layout:
    # Left panel: Video (640x480) + Angles (640x160) = 640x640
    # Right panel: EMG plots (400x640)
    display_width = 640 + 400
    cv2.resizeWindow('Replay', display_width, display_height)  # Initial window size, adjustable

    # Initialize playback timing
    num_frames = len(frames)
    current_frame_idx = 0
    playback_start_time = time.perf_counter()  # High-resolution start time
    playback_start_timestamp = frames[0][0]

    while current_frame_idx < num_frames:
        # Calculate the elapsed playback time adjusted by playback_speed
        elapsed_real_time = (time.perf_counter() - playback_start_time) * playback_speed
        target_timestamp = playback_start_timestamp + elapsed_real_time

        # Get the closest frame index based on the current playback time
        closest_frame_idx = playback.get_closest_frame_index(target_timestamp)
        if closest_frame_idx == -1:
            print("No frames available.")
            break

        # Avoid processing the same frame repeatedly
        if closest_frame_idx != current_frame_idx:
            current_frame_idx = closest_frame_idx

        frame_timestamp, frame_path = frames[current_frame_idx]

        # Load the frame image
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Failed to load frame: {frame_path}")
            current_frame_idx += 1
            continue

        # Resize and crop the frame to 640x480
        frame_resized = cv2.resize(frame, (640, 480))

        # Convert the image to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Process the image and detect hands
        results = hands.process(frame_rgb)

        # Draw hand landmarks on the frame
        frame_with_landmarks = draw_hand_landmarks(frame_resized.copy(), results.multi_hand_landmarks, mp_hands, mp_drawing)

        # Get EMG data closest to the current frame timestamp
        closest_emg = playback.get_closest_emg_data(frame_timestamp)

        # Get EMG window for plotting (past data only)
        emg_window_df = playback.get_emg_window(frame_timestamp)

        # Get finger angles for the current frame
        finger_angles = playback.get_closest_finger_angles(frame_timestamp)

        # Create EMG plots without highlighting all closest points
        if not emg_window_df.empty:
            emg_plot_img = create_emg_plots(emg_window_df, playback.emg_channels, plot_height=640, plot_width=400)
        else:
            # Create a blank image if there's no EMG data
            emg_plot_img = np.zeros((640, 400, 3), dtype='uint8')

        # Create Finger Angles display (only 5 fingers)
        angles_display_img = create_finger_angles_display(finger_angles, playback.finger_names, display_size=(640, 160))

        # Resize components based on the desired display height
        scale_factor = display_height / 640.0  # Base height is 640 pixels

        # Scale video frame
        frame_scaled = cv2.resize(frame_with_landmarks, (int(640 * scale_factor), int(480 * scale_factor)))

        # Scale angles display
        angles_display_scaled = cv2.resize(angles_display_img, (int(640 * scale_factor), int(160 * scale_factor)))

        # Combine video frame and angles display vertically
        combined_left_panel = np.vstack((frame_scaled, angles_display_scaled))

        # Scale EMG plots
        emg_plot_scaled = cv2.resize(emg_plot_img, (int(400 * scale_factor), int(640 * scale_factor)))

        # Concatenate the combined left panel and EMG plots horizontally
        composite_frame = np.hstack((combined_left_panel, emg_plot_scaled))

        # Display the composite frame
        cv2.imshow('Replay', composite_frame)

        # Handle keypress
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit
            print("Exiting replay.")
            break

        # Increment frame index
        current_frame_idx += 1

    # Release resources
    hands.close()
    cv2.destroyAllWindows()
    print("Replay finished.")

# =======================
# Entry Point
# =======================

if __name__ == "__main__":
    main()

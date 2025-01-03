# brainflow_raw_server.py
import logging
import sys
import numpy as np
import datetime
import threading
import time
import uuid  # For unique source_id

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from pylsl import StreamInfo, StreamOutlet, local_clock

import serial.tools.list_ports
from scipy.signal import butter, sosfilt, sosfilt_zi, iirnotch, tf2sos


def find_serial_port():
    """
    Automatically detect the serial port for the OpenBCI board.
    Returns the port name as a string, or raises an exception if not found.
    """
    ports = serial.tools.list_ports.comports()
    for port in ports:
        # Check for specific keywords or VID/PID
        if 'OpenBCI' in port.description or 'usbserial' in port.device.lower():
            return port.device
    logging.warning("OpenBCI board not found. Using synthetic board instead.")
    return ''


class BrainFlowServer:
    def __init__(self, board_id:BoardIds=BoardIds.SYNTHETIC_BOARD, stream_raw:bool=False,
                 stream_filtered:bool=False, enable_bandpass:bool=False, 
                 enable_notch=False, lowpass_fs: float = None,highpass_fs: float = 10, 
                 filter_order: int = 4, openbci_emg: bool = False, openbci_eeg: bool = False, 
                 serial_port: str = '', update_fps: int = 0, channel_list:list[int]=None, 
                 notch_freqs:list[int]=[60], quality_factor:float=50):
        """
        Initializes the BrainFlow server for streaming raw data.

        Args:
            board_id (int): Board ID (default: BoardIds.SYNTHETIC_BOARD).
            lowpass_fs (float): Lowpass cutoff frequency in Hz (default: None = Sampling Rate / 2).
            highpass_fs (float): Highpass cutoff frequency in Hz.
            filter_order (int): Filter order (default: 4).
            openbci_emg (bool): Enable OpenBCI EMG configuration.
            openbci_eeg (bool): Enable OpenBCI EEG configuration.
            serial_port (str): Serial port for the OpenBCI board.
            update_fps (int): Update rate in FPS (default: 0 = sampling rate).
            channel_list (list): List of channel IDs (default: None).
            notch_freqs (list or None): List of frequencies in Hz to apply notch filters.
                                        Defaults to [60] if None.
            quality_factor (float): Quality factor for notch filters. Higher Q means narrower notch.
        """
        # Initialize logging specific to this module
        self._init_logging()

        self.params = BrainFlowInputParams()
        self.params.serial_port = serial_port
        self.board_id = board_id
        self.openbci = openbci_emg or openbci_eeg
        self.stream_raw = stream_raw
        self.stream_filtered = stream_filtered

        # Initialize the board
        try:
            self.board_shim = BoardShim(self.board_id, self.params)
            logging.info(f"Initializing BoardShim with board_id: {self.board_id}")
            self.board_shim.prepare_session()
            logging.info("Board session prepared successfully.")
        except BrainFlowError as e:
            logging.error(f"Failed to prepare session: {e}")
            raise e
        except Exception as e:
            logging.error(f"Unexpected error during board initialization: {e}")
            raise e

        # Retrieve EXG channels and sampling rate
        try:
            if channel_list is not None:
                self.exg_channels = channel_list
            else:
                self.exg_channels = BoardShim.get_exg_channels(self.board_id)
            self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
            self.timestamp_channel = BoardShim.get_timestamp_channel(self.board_id)
            self.num_channels = len(self.exg_channels)
            logging.info(f"EXG Channels: {self.exg_channels}")
            logging.info(f"Sampling Rate: {self.sampling_rate} Hz")
        except BrainFlowError as e:
            logging.error(f"Failed to retrieve board information: {e}")
            self.release_board()
            raise e

        # Disable board logger
        try:
            self.board_shim.disable_board_logger()
            logging.info("Board logger disabled.")
        except BrainFlowError as e:
            logging.error(f"Failed to disable board logger: {e}")
            self.release_board()
            raise e

        # Configure board if EMG is enabled
        if self.openbci:
            try:
                # Configuration parameters
                POWER_DOWN = 0  # EMG is 0 = Normal operation
                GAIN_SET = 6    # EMG is 6 = Gain 24
                INPUT_TYPE_SET = 0  # EMG is 0 = Normal electrode input
                BIAS_SET = 1    # EMG is 1
                SRB2_SET = 0 if openbci_emg else 1
                SRB1_SET = 0

                # Channel names: 1 2 3 4 5 6 7 8 Q W E R T Y U I
                channel_names = ['1', '2', '3', '4', '5', '6', '7', '8', 'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I']

                # Ensure each value is a single digit
                def valid_value(value):
                    return len(str(value)) == 1

                if not all(map(valid_value, [POWER_DOWN, GAIN_SET, INPUT_TYPE_SET, BIAS_SET, SRB2_SET, SRB1_SET])):
                    raise ValueError("Invalid value for config settings (All must be a single character).")

                # Set config settings for each channel
                config_list = [
                    f"x{channel_names[channel - 1]}{POWER_DOWN}{GAIN_SET}{INPUT_TYPE_SET}{BIAS_SET}{SRB2_SET}{SRB1_SET}X"
                    for channel in self.exg_channels
                ]
                config = ''.join(config_list)
                if config:
                    self.board_shim.config_board(config)
                    logging.info("Configured settings for EMG.")
            except BrainFlowError as e:
                logging.error(f"Failed to configure board for EMG: {e}")
                self.release_board()
                raise e
            except Exception as e:
                logging.error(f"Unexpected error during EMG configuration: {e}")
                self.release_board()
                raise e

        # Start streaming
        stream_buffer_size = int(self.sampling_rate * 5)  # 5 seconds of data
        try:
            self.board_shim.start_stream(stream_buffer_size, '')
            logging.info("Board streaming started successfully.")
        except BrainFlowError as e:
            logging.error(f"Failed to start board streaming: {e}")
            self.release_board()
            raise e

        # Initialize LSL streams
        self._init_lsl_stream()

        # Set up update parameters
        if update_fps == 0:
            self.update_fps = self.sampling_rate  # Update at the same rate as sampling
        else:
            self.update_fps = update_fps
        self.update_interval = 1 / self.update_fps
        self.time_diff = local_clock() - time.time()

        # Control variables for the update loop
        self.running = False
        self.update_thread = None
        
        self.enable_bandpass = enable_bandpass
        self.enable_notch = enable_notch

        if enable_bandpass:
            if lowpass_fs is None:
                lowpass_fs = self.sampling_rate / 2

            self.sos_band = self.design_bandpass_filter(lowpass_fs, highpass_fs, fs=self.sampling_rate,
                                                        order=filter_order)
            n_sections = self.sos_band.shape[0]
            # Initialize filter states for each channel
            # sosfilt_zi returns (n_sections, 2)
            zi_single_band = sosfilt_zi(self.sos_band)  # Shape: (n_sections, 2)
            # Tile for each channel: (num_channels, n_sections, 2)
            self.zi_band = np.tile(zi_single_band, (self.num_channels, 1, 1))
            # Optionally, scale initial conditions if needed
            self.zi_band *= 0  # Initialize to zero or any desired initial condition
            logging.info("Bandpass filter designed and initialized.")
            
        if enable_notch:
            # Design notch filters
            self.notch_freqs = notch_freqs if notch_freqs is not None else [60]
            self.sos_notch = []
            self.zi_notch = []

            if self.notch_freqs:
                for freq in self.notch_freqs:
                    sos = self.design_notch_filter(freq, fs=self.sampling_rate, quality_factor=quality_factor)
                    self.sos_notch.append(sos)
                    # Initialize filter states for each channel and each notch filter
                    zi_single_notch = sosfilt_zi(sos)  # Shape: (n_sections, 2)
                    # Tile for each channel: (num_channels, n_sections, 2)
                    zi_channel_notch = np.tile(zi_single_notch, (self.num_channels, 1, 1))
                    # Initialize to zero or any desired initial condition
                    self.zi_notch.append(zi_channel_notch)
                logging.info(f"Designed notch filters for frequencies: {self.notch_freqs}")

        logging.info("Initialized BrainFlowServer for raw streaming.")

    def _init_logging(self):
        """
        Configures the logging settings.
        """
        logging.basicConfig(
            level=logging.DEBUG,  # Set to DEBUG for detailed logs
            format='[%(asctime)s] [%(levelname)s] %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        logging.info("Logging initialized for BrainFlowServer.")

    def _init_lsl_stream(self):
        """
        Initializes LSL streams for raw data.
        """
        if self.stream_raw:
            # Define Raw LSL stream info with a unique source_id
            self.raw_lsl_info = StreamInfo(
                name="raw_exg",
                type="EXG",
                channel_count=len(self.exg_channels),
                nominal_srate=self.sampling_rate,
                channel_format="float32",
                source_id=f"BrainFlowLSLStream_{uuid.uuid4()}"  # Unique identifier
            )
            self.raw_lsl_outlet = StreamOutlet(self.raw_lsl_info)
            logging.info("Raw LSL stream initialized with unique source ID.")
        if self.stream_filtered:
            self.filtered_lsl_info = StreamInfo(
                name="filtered_exg",
                type="EXG",
                channel_count=len(self.exg_channels),
                nominal_srate=self.sampling_rate,
                channel_format="float32",
                source_id=f"BrainFlowLSLStream_filtered_{uuid.uuid4()}"  # Unique identifier
            )
            self.filtered_lsl_outlet = StreamOutlet(self.filtered_lsl_info)
            logging.info("Filtered LSL stream initialized with unique source ID.")

    @staticmethod
    def design_bandpass_filter(lowpass_fs, highpass_fs, fs, order=4):
        """
        Design a Butterworth bandpass filter using Second-Order Sections (SOS).

        Args:
            lowpass_fs (float): Low cutoff frequency in Hz.
            highpass_fs (float): High cutoff frequency in Hz.
            fs (float): Sampling frequency in Hz.
            order (int): Filter order (default: 4).

        Returns:
            sos (ndarray): Second-Order Sections representation of the filter.
        """
        nyquist = 0.5 * fs
        Wn = [highpass_fs / nyquist, lowpass_fs / nyquist]

        # Ensure cutoff frequencies are within valid range
        Wn[0] = max(Wn[0], 0.00000001)  # Ensure low cutoff is at least 0.00...1% of Nyquist
        Wn[1] = min(Wn[1], 0.99999999)  # Ensure high cutoff is at most 99.99...9% of Nyquist
        sos = butter(order, Wn=Wn, btype='band', analog=False, output='sos')
        return sos

    @staticmethod
    def design_notch_filter(freq, fs, quality_factor=50):
        """
        Design a notch filter using Second-Order Sections (SOS).

        Args:
            freq (float): Frequency to notch out in Hz.
            fs (float): Sampling frequency in Hz.
            quality_factor (float): Quality factor for the notch filter.

        Returns:
            sos (ndarray): Second-Order Sections representation of the notch filter.
        """
        nyquist = 0.5 * fs
        w0 = freq / nyquist
        w0 = max(w0, 0.00000001)  # Ensure w0 is at least 0.00...1% of Nyquist
        w0 = min(w0, 0.99999999)  # Ensure w0 is at most 99.99...9% of Nyquist
        b, a = iirnotch(w0, quality_factor)
        sos = tf2sos(b, a)
        return sos

    @staticmethod
    def apply_bandpass(data, sos, zi):
        """
        Apply bandpass filtering to multi-channel data in real-time using sosfilt.

        Args:
            data (ndarray): New data batch (num_channels, batch_size).
            sos (ndarray): Second-Order Sections representation of the filter.
            zi (ndarray): Initial filter states (num_channels, n_sections, 2).

        Returns:
            filtered_data (ndarray): Filtered data.
            updated_zi (ndarray): Updated filter states.
        """
        num_channels = data.shape[0]
        filtered_data = np.empty_like(data)
        updated_zi = np.empty_like(zi)

        for ch in range(num_channels):
            # Apply sosfilt to each channel with its own state
            filtered_data[ch], updated_zi[ch] = sosfilt(sos, data[ch], zi=zi[ch])

        return filtered_data, updated_zi
    
    @staticmethod
    def apply_notch(data, sos_list, zi_list):
        """
        Apply multiple notch filters to multi-channel data in real-time using sosfilt.

        Args:
            data (ndarray): Data to be filtered (num_channels, batch_size).
            sos_list (list of ndarray): List of SOS arrays for each notch filter.
            zi_list (list of ndarray): List of initial filter states for each notch filter.

        Returns:
            filtered_data (ndarray): Notch filtered data.
            updated_zi_list (list of ndarray): Updated filter states for each notch filter.
        """
        filtered_data = data.copy()
        updated_zi_list = []

        for sos, zi in zip(sos_list, zi_list):
            num_channels = filtered_data.shape[0]
            temp_filtered = np.empty_like(filtered_data)
            temp_zi = np.empty_like(zi)

            for ch in range(num_channels):
                temp_filtered[ch], temp_zi[ch] = sosfilt(sos, filtered_data[ch], zi=zi[ch])

            filtered_data = temp_filtered
            updated_zi_list.append(temp_zi)

        return filtered_data, updated_zi_list
    
    def apply_filter(self, data):
        """
        Apply bandpass and notch filters to multi-channel data in real-time using sosfilt.

        Args:
            data (ndarray): Data to be filtered (num_channels, batch_size).
            sos_band (ndarray): SOS array for bandpass filter.
            sos_notch (list of ndarray): List of SOS arrays for each notch filter.
            zi_band (ndarray): Initial filter states for bandpass filter.
            zi_notch (list of ndarray): List of initial filter states for each notch filter.

        Returns:
            filtered_data (ndarray): Filtered data.
            updated_zi_band (ndarray): Updated filter states for bandpass filter.
            updated_zi_notch (list of ndarray): Updated filter states for each notch filter.
        """
        data = data.copy()
        # Apply bandpass filter
        if self.enable_bandpass:
            data, self.zi_band = self.apply_bandpass(data, self.sos_band, self.zi_band)
        # Apply notch filters
        if self.enable_notch:
            data, self.zi_notch = self.apply_notch(data, self.sos_notch, self.zi_notch)
        return data

    def update(self):
        """
        Fetches new data from the board and streams via LSL.
        """
        try:
            # Fetch the latest samples
            new_data = self.board_shim.get_board_data()
            if new_data.size == 0:
                logging.debug('No new data')
                return False

            # Extract EXG channels and timestamps
            raw_data = new_data[self.exg_channels, :]
            timestamp_data = new_data[self.timestamp_channel, :] + self.time_diff
            logging.debug(f'Raw shape: {raw_data.shape}')
            if self.stream_raw:
                self.raw_lsl_outlet.push_chunk(raw_data.T.tolist(), timestamp_data.T.tolist())

            # Filtered data
            if self.stream_filtered and (self.enable_bandpass or self.enable_notch):
                filtered_data = self.apply_filter(raw_data)
                
                self.filtered_lsl_outlet.push_chunk(filtered_data.T.tolist(), timestamp_data.T.tolist())

            return True

        except BrainFlowError as e:
            logging.error(f"BrainFlowError in update loop: {e}")
        except Exception as e:
            logging.error(f"Unexpected error in update loop: {e}")
        # Return False if an error occurred
        return False

    def update_loop(self):
        """
        Continuously fetches new data from the board and updates the LSL streams using absolute scheduling.
        """
        self.running = True
        logging.info("Starting update loop.")

        next_call = time.perf_counter()

        while self.running:
            update_success = self.update()

            # Schedule next call
            if update_success:
                next_call += self.update_interval

            sleep_time = next_call - time.perf_counter()

            if sleep_time > 0:
                time.sleep(sleep_time)

        logging.info("Update loop stopped.")

    def start(self):
        """
        Starts the update loop in a separate thread.
        """
        self.update_thread = threading.Thread(target=self.update_loop, daemon=True)
        self.update_thread.start()
        logging.info("Update thread started.")

    def stop(self):
        """
        Stops the update loop and releases board resources.
        """
        self.running = False
        if self.update_thread is not None:
            self.update_thread.join()
        self.release_board()
        logging.info("Server stopped and resources released.")

    def release_board(self):
        """
        Releases board resources.
        """
        try:
            if self.board_shim.is_prepared():
                logging.info('Stopping and releasing board session.')
                self.board_shim.stop_stream()
                self.board_shim.release_session()
        except Exception as e:
            logging.error(f"Error releasing board: {e}")


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.WARN,  # Set to DEBUG for detailed logs
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Enable BrainFlow's internal logger
    BoardShim.enable_dev_board_logger()

    serial_port = find_serial_port()
    board_id = BoardIds.SYNTHETIC_BOARD  # Change as needed, e.g., BoardIds.OPENBCI

    # Initialize the BrainFlow server
    server = BrainFlowServer(
        board_id=board_id,
        openbci_emg=True,  # Set to True if using EMG
        stream_filtered=True,
        serial_port=serial_port,
        update_fps=0,  #
        channel_list=[1, 2, 3, 4, 5, 6, 7, 8],  # Adjust channels as needed
        enable_bandpass=True,
        enable_notch=True,
        notch_freqs=[60],  # Default notch frequency set to 60 Hz
        quality_factor=50  # Adjust quality factor as needed
    )

    try:
        server.start()
        logging.info("BrainFlowServer is running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)  # Keep the main thread alive
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt detected. Stopping server.")
    finally:
        server.stop()


if __name__ == '__main__':
    main()

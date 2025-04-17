import logging
import sys
import time
import numpy as np
import datetime
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from pylsl import StreamInfo, StreamOutlet, local_clock
import serial.tools.list_ports

from iir import IIR

# TODO: MAKE FILTERING OPTIONAL

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
    def __init__(self, fps=None, board_id=BoardIds.SYNTHETIC_BOARD, is_emg=False, 
                 serial_port='', lsl_raw=False, lsl_filtered=True,
                 include_channels=None):
        self._init_logging()

        self.params = BrainFlowInputParams()
        self.params.serial_port = serial_port
        self.board_id = board_id
        self.is_emg = is_emg

        # Initialize the board
        try:
            self.board_shim = BoardShim(self.board_id, self.params)
            logging.info(f"Initializing BoardShim with board_id: {self.board_id}")
            self.board_shim.prepare_session()
            logging.info("Board session prepared successfully.")
        except BrainFlowError as e:
            logging.error(f"Failed to prepare session: {e}")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Unexpected error during board initialization: {e}")
            sys.exit(1)

        # Retrieve EXG channels and sampling rate
        try:
            if include_channels:
                self.exg_channels = include_channels
            else:
                self.exg_channels = BoardShim.get_exg_channels(self.board_id)
            self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
            self.timestamp_channel = BoardShim.get_timestamp_channel(self.board_id)
            logging.info(f"EXG Channels: {self.exg_channels}")
            logging.info(f"Sampling Rate: {self.sampling_rate} Hz")
        except BrainFlowError as e:
            logging.error(f"Failed to retrieve board information: {e}")
            self.release_board()
            sys.exit(1)
            
                
        if fps is None:
            fps = self.sampling_rate
        
        self.fps = fps

        # Configure board if EMG is enabled
        if self.is_emg and self.board_id in (BoardIds.CYTON_BOARD, BoardIds.CYTON_DAISY_BOARD):
            try:
                # Configuration parameters
                POWER_DOWN = 0  # EMG is 0 = Normal operation
                GAIN_SET = 6    # EMG is 6 = Gain 24
                INPUT_TYPE_SET = 0  # EMG is 0 = Normal electrode input
                BIAS_SET = 1    # EMG is 1
                SRB2_SET = 0    # EMG is 0
                SRB1_SET = 0    # EMG is 0
                
                # Channel names: 1 2 3 4 5 6 7 8 Q W E R T Y U I
                channel_names = ['1', '2', '3', '4', '5', '6', '7', '8', 'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I']
                # Ensure each value is a single digit
                def valid_value(value):
                    return len(str(value)) == 1

                if not all(map(valid_value, [POWER_DOWN, GAIN_SET, INPUT_TYPE_SET, BIAS_SET, SRB2_SET, SRB1_SET])):
                    raise ValueError("Invalid value for config settings (All must be a single character).")

                # Set config settings for each channel
                config_list = [f"x{channel_names[channel-1]}{POWER_DOWN}{GAIN_SET}{INPUT_TYPE_SET}{BIAS_SET}{SRB2_SET}{SRB1_SET}X" 
                               for channel in self.exg_channels]
                config = ''.join(config_list)
                if config:
                    self.board_shim.config_board(config)
                    logging.info("Configured settings for EMG.")
            except BrainFlowError as e:
                logging.error(f"Failed to configure board for EMG: {e}")
                self.release_board()
                sys.exit(1)

        # Start streaming
        stream_buffer_size = int(self.sampling_rate * 5)  # 5 seconds of data
        try:
            self.board_shim.start_stream(stream_buffer_size, '')
            logging.info("Board streaming started successfully.")
        except BrainFlowError as e:
            logging.error(f"Failed to start board streaming: {e}")
            self.release_board()
            sys.exit(1)

        # Set the buffer size based on the window length (5 seconds)
        self.buffer_size = int(self.sampling_rate * 5)

        # Initialize LSL streams
        self.lsl_raw = lsl_raw
        self.lsl_filtered = lsl_filtered
        self._init_lsl_streams()

        # Initialize data buffers
        self.raw_exg_buffer = np.zeros((len(self.exg_channels), self.buffer_size))
        self.filtered_exg_buffer = np.zeros((len(self.exg_channels), self.buffer_size))
        self.timestamp_buffer = np.zeros(self.buffer_size)

        # Compute time difference for LSL timestamps
        self.time_diff = local_clock() - datetime.datetime.now().timestamp()

        # Initialize the IIR filter:
        #   - lowpass_fs is set to half the sampling rate (i.e. the upper cutoff)
        #   - highpass_fs is set to 10.0 Hz (i.e. the lower cutoff)
        #   - notch_fs_list is set to remove 60 Hz interference.
        self.iir_filter = IIR(
            num_channels=len(self.exg_channels),
            fs=self.sampling_rate,
            lowpass_fs=self.sampling_rate / 2,
            highpass_fs=10.0,
            notch_fs_list=[60],
            filter_order=4
        )

    def _init_logging(self):
        """
        Configures the logging settings.
        """
        logging.basicConfig(
            level=logging.DEBUG,
            format='[%(asctime)s] [%(levelname)s] %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        logging.info("Logging initialized for BrainFlowServer.")

    def _init_lsl_streams(self):
        """
        Initializes LSL streams for raw and filtered data.
        """
        if self.lsl_raw:
            self.raw_lsl_info = StreamInfo(
                name="raw_exg",
                type="EXG",
                channel_count=len(self.exg_channels),
                nominal_srate=self.sampling_rate,
                channel_format="float32",
                source_id="BrainFlowLSLStream"
            )
            self.raw_lsl_outlet = StreamOutlet(self.raw_lsl_info)
            logging.info("Raw LSL initialized.")

        if self.lsl_filtered:
            self.filtered_lsl_info = StreamInfo(
                name="filtered_exg",
                type="EXG",
                channel_count=len(self.exg_channels),
                nominal_srate=self.sampling_rate,
                channel_format="float32",
                source_id="BrainFlowLSLStream"
            )
            self.filtered_lsl_outlet = StreamOutlet(self.filtered_lsl_info)
            logging.info("Filtered LSL initialized.")

    def update_lsl_streams(self, num_samples):
        """
        Pushes the latest data chunks to the LSL streams.
        
        :param num_samples: Number of new samples to push
        """
        if num_samples == 0:
            return

        raw_data = self.raw_exg_buffer[:, -num_samples:].copy()
        filtered_data = self.filtered_exg_buffer[:, -num_samples:].copy()
        timestamp_data = self.timestamp_buffer[-num_samples:].copy()

        if self.lsl_raw:
            self.raw_lsl_outlet.push_chunk(raw_data.T.tolist(), timestamp_data.tolist())

        if self.lsl_filtered:
            self.filtered_lsl_outlet.push_chunk(filtered_data.T.tolist(), timestamp_data.tolist())

    def update(self):
        """
        Fetches new data from the board, processes it using the IIR filter, 
        and streams via LSL.
        """
        try:
            # Fetch the latest samples from the board
            new_data = self.board_shim.get_board_data()
            if new_data is None or new_data.size == 0:
                return

            num_samples = new_data.shape[1]
            exg_data = new_data[self.exg_channels, :]  # shape: (channels, num_samples)
            timestamp_data = new_data[self.timestamp_channel, :] + self.time_diff

            # Update timestamp buffer
            self.timestamp_buffer = np.roll(self.timestamp_buffer, -num_samples)
            self.timestamp_buffer[-num_samples:] = timestamp_data

            # Update raw data buffer
            self.raw_exg_buffer = np.roll(self.raw_exg_buffer, -num_samples, axis=1)
            self.raw_exg_buffer[:, -num_samples:] = exg_data

            # Apply IIR filtering to the new data chunk:
            #   - Transpose data so rows=samples, columns=channels.
            new_chunk = exg_data.T  # shape: (num_samples, num_channels)
            filtered_chunk = self.iir_filter.apply_inplace(new_chunk)
            # Store the filtered chunk back in buffer (transpose back to channels x samples)
            self.filtered_exg_buffer = np.roll(self.filtered_exg_buffer, -num_samples, axis=1)
            self.filtered_exg_buffer[:, -num_samples:] = filtered_chunk.T

            # Update LSL streams with new data
            self.update_lsl_streams(num_samples)

        except BrainFlowError as e:
            logging.error(f"BrainFlowError in update function: {e}")
            self.release_board()
            sys.exit(1)
        except Exception as e:
            logging.error(f"Unexpected error in update function: {e}")
            self.release_board()
            sys.exit(1)

    def run(self):
        """
        Continuously updates the board data until interrupted.
        """
        logging.info("Starting data acquisition loop. Press Ctrl+C to stop.")
        try:
            while True:
                self.update()
                time.sleep(1/self.fps)  # ~50 Hz update rate
        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt received, stopping data acquisition.")
            self.cleanup()

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

    def cleanup(self):
        """
        Releases the board and performs cleanup.
        """
        self.release_board()
        logging.info("Cleanup complete. Exiting.")

def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Enable BrainFlow's internal logger if needed
    BoardShim.enable_dev_board_logger()

    try:
        serial_port = find_serial_port()
        board_id = BoardIds.CYTON_BOARD
        is_emg = True

        brainflow_graph = BrainFlowServer(board_id=board_id, is_emg=is_emg, serial_port=serial_port,
                                         lsl_raw=True, lsl_filtered=True,
                                         include_channels=[1, 2, 3, 4, 5, 6, 7, 8])
        brainflow_graph.run()
    except RuntimeError as e:
        logging.error(f"RuntimeError: {e}")
        sys.exit(1)
    except BrainFlowError as e:
        logging.error(f"BrainFlowError: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unhandled exception: {e}")
        sys.exit(1)
    finally:
        logging.info('Program terminated.')

if __name__ == '__main__':
    main()

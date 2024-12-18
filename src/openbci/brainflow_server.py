# graph_brainflow.py

import logging
import sys
import numpy as np
import datetime
from PyQt5 import QtCore

from brainflow import NoiseTypes
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from pylsl import StreamInfo, StreamOutlet, local_clock

from graph import BaseGraph

import serial.tools.list_ports

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

class BrainFlowGraph(BaseGraph):
    def __init__(self, board_id=BoardIds.SYNTHETIC_BOARD, is_emg=False, 
                 serial_port='', lsl_raw=False, lsl_filtered=True,
                 include_channels=None):
        # Initialize logging specific to this module
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
                config_list = [f"x{channel_names[channel-1]}{POWER_DOWN}{GAIN_SET}{INPUT_TYPE_SET}{BIAS_SET}{SRB2_SET}{SRB1_SET}X" for channel in self.exg_channels]
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

        # Initialize the base graph
        super().__init__(
            title='BrainFlow Plot',
            window_size=5,
            sampling_rate=self.sampling_rate,
            num_channels=len(self.exg_channels)
        )

        # Initialize LSL streams
        self.lsl_raw = lsl_raw
        self.lsl_filtered = lsl_filtered
        self._init_lsl_streams()

        # Initialize data buffers
        self.raw_exg_buffer = np.zeros((len(self.exg_channels), self.buffer_size))
        self.filtered_exg_buffer = np.zeros((len(self.exg_channels), self.buffer_size))
        self.timestamp_buffer = np.zeros(self.buffer_size)

        # Set up a timer to call the update function periodically
        self.update_speed_ms = 1000 // 50  # 20 ms for ~50 FPS
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(self.update_speed_ms)
        self.time_diff = local_clock() - datetime.datetime.now().timestamp()

        logging.info("Entering the Qt event loop.")
        self.app.exec_()

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
        logging.info("Logging initialized for BrainFlowGraph.")

    def _init_lsl_streams(self):
        """
        Initializes LSL streams for raw and filtered data.
        """
        # Define Raw LSL stream info
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

        # Define Filtered LSL stream info
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
        Fetches new data from the board, processes it, updates plots, and streams via LSL.
        """
        try:
            # Fetch the latest samples
            new_data = self.board_shim.get_board_data()
            if new_data is None or new_data.size == 0:
                return

            # Extract EXG channels
            num_samples = new_data.shape[1]
            exg_data = new_data[self.exg_channels, :]  # shape: (channels, samples)
            timestamp_data = new_data[self.timestamp_channel, :] + self.time_diff

            # Update timestamp buffer
            self.timestamp_buffer = np.roll(self.timestamp_buffer, -num_samples)
            self.timestamp_buffer[-num_samples:] = timestamp_data

            # Update raw data buffer
            self.raw_exg_buffer = np.roll(self.raw_exg_buffer, -num_samples, axis=1)
            self.raw_exg_buffer[:, -num_samples:] = exg_data

            # Roll the filtered data buffer
            self.filtered_exg_buffer = np.roll(self.filtered_exg_buffer, -num_samples, axis=1)
            
            # Apply filters and update filtered buffer
            for count, channel_data in enumerate(self.raw_exg_buffer):
                # Grab the latest data
                channel_data = channel_data.copy()

                # Detrend
                DataFilter.detrend(channel_data, DetrendOperations.CONSTANT.value)

                # Bandpass filter
                DataFilter.perform_bandpass(
                    channel_data, self.sampling_rate, 10.0, self.sampling_rate/2, 4, FilterTypes.BUTTERWORTH.value, 1.0
                )
                
                # Bandstop filter
                DataFilter.perform_bandstop(
                    channel_data, self.sampling_rate, 58.0, 62.0, 4, FilterTypes.BUTTERWORTH.value, 1.0
                )

                # Remove environmental noise
                DataFilter.remove_environmental_noise(
                    channel_data, self.sampling_rate, NoiseTypes.FIFTY_AND_SIXTY.value
                )

                # Update filtered data buffer with new samples
                self.filtered_exg_buffer[count, -num_samples:] = channel_data[-num_samples:]

                # Update plot
                self.curves[count].setData(self.filtered_exg_buffer[count])

            # Update LSL streams
            self.update_lsl_streams(num_samples)

        except BrainFlowError as e:
            logging.error(f"BrainFlowError in update function: {e}")
            self.release_board()
            sys.exit(1)
        except Exception as e:
            logging.error(f"Unexpected error in update function: {e}")
            self.release_board()
            sys.exit(1)

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
        Overrides the cleanup method to release the board.
        """
        self.release_board()

def main():
    # Configure logging (if not already configured in BaseGraph)
    logging.basicConfig(
        level=logging.DEBUG,  # Set to DEBUG for detailed logs
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Enable BrainFlow's internal logger
    BoardShim.enable_dev_board_logger()

    try:
        serial_port = find_serial_port()
        board_id = BoardIds.SYNTHETIC_BOARD
        is_emg = True

        brainflow_graph = BrainFlowGraph(board_id=board_id, is_emg=is_emg, serial_port=serial_port,
                                         lsl_raw=False, lsl_filtered=True,
                                         include_channels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
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

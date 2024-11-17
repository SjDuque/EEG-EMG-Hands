import argparse
import logging
import pyqtgraph as pg
from brainflow import NoiseTypes
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from PyQt5 import QtWidgets, QtCore 
from PyQt5.QtWidgets import QShortcut
from PyQt5.QtGui import QKeySequence

import serial.tools.list_ports
import sys
import time
import threading
import queue

from pylsl import StreamInfo, StreamOutlet, local_clock

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
    raise RuntimeError("OpenBCI board not found. Ensure it is connected and drivers are installed.")

class CustomGraphicsLayoutWidget(pg.GraphicsLayoutWidget):
    """
    Subclass of GraphicsLayoutWidget to handle key press events.
    """
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

class LSLStreamer(threading.Thread):
    """
    A separate thread to handle LSL streaming without blocking the main GUI thread.
    """
    def __init__(self, outlet, data_queue, stop_event):
        super().__init__()
        self.outlet = outlet
        self.data_queue = data_queue
        self.stop_event = stop_event

    def run(self):
        logging.info("LSLStreamer thread started.")
        while not self.stop_event.is_set():
            try:
                # Wait for new data with a timeout to allow checking the stop_event
                sample = self.data_queue.get(timeout=0.1)
                if sample is None:
                    continue  # Skip if sample is None
                timestamp = local_clock()
                self.outlet.push_sample(sample, timestamp)
            except queue.Empty:
                continue  # No data received, continue checking
            except Exception as e:
                logging.error(f"Error in LSLStreamer thread: {e}", exc_info=True)
        logging.info("LSLStreamer thread terminated.")

class Graph:
    def __init__(self, board_id=BoardIds.SYNTHETIC_BOARD, emg=False, serial_port=''):
        self.params = BrainFlowInputParams()
        self.params.serial_port = serial_port
        self.board_id = board_id
        self.emg = emg

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
            self.exg_channels = BoardShim.get_exg_channels(self.board_id)
            self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
            logging.info(f"EXG Channels: {self.exg_channels}")
            logging.info(f"Sampling Rate: {self.sampling_rate} Hz")
        except BrainFlowError as e:
            logging.error(f"Failed to retrieve board information: {e}")
            self.release_board()
            sys.exit(1)

        # Configure board if EMG is enabled
        if self.emg and self.board_id in (BoardIds.CYTON_BOARD, BoardIds.CYTON_DAISY_BOARD):
            try:
                config_srb2_list = [f"x{channel}060100" for channel in self.exg_channels]
                config_srb2 = ''.join(config_srb2_list)
                if config_srb2:
                    self.board_shim.config_board(config_srb2)
                    logging.info("Configured SRB2 settings for EMG.")
            except BrainFlowError as e:
                logging.error(f"Failed to configure board for EMG: {e}")
                self.release_board()
                sys.exit(1)

        # Start streaming
        try:
            self.board_shim.start_stream(450000, '')
            logging.info("Board streaming started successfully.")
        except BrainFlowError as e:
            logging.error(f"Failed to start board streaming: {e}")
            self.release_board()
            sys.exit(1)

        # Initialize LSL stream
        self._init_lsl()

        # Initialize threading components
        self.data_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.lsl_thread = LSLStreamer(self.outlet, self.data_queue, self.stop_event)
        self.lsl_thread.start()
        self.update_speed_ms = 1000 // 250 # 250 Hz
        self.window_size = 5                # 1 seconds of data
        self.num_points = self.window_size * self.sampling_rate

        self.app = QtWidgets.QApplication([])

        # Use the custom GraphicsLayoutWidget
        self.win = CustomGraphicsLayoutWidget(title='BrainFlow Plot and LSL Stream')
        self.win.resize(800, 600)
        self.win.setWindowTitle('BrainFlow Plot and LSL Stream')
        self.win.show()

        # Ensure that board is released when window is closed
        self.win.closeEvent = self.on_close

        self._init_timeseries()

        # Optionally, add a shortcut for ESC key as an alternative
        shortcut = QShortcut(QKeySequence("Esc"), self.win)
        shortcut.activated.connect(self.win.close)

        # Initialize sample tracking
        self.last_sample_index = 0

        # Set up a timer to call the update function periodically
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(self.update_speed_ms)

        logging.info("Entering the Qt event loop.")
        self.app.exec_()

    def _init_lsl(self):
        """Initializes the LSL stream for streaming EEG/EMG data."""
        try:
            stream_name = 'OpenBCI_EMG_Stream'
            stream_type = 'EMG'
            channel_count = len(self.exg_channels)
            nominal_srate = self.sampling_rate  # Hz
            channel_format = 'float32'
            source_id = f'OpenBCI_EMG_{int(time.time())}'

            info = StreamInfo(name=stream_name, type=stream_type, channel_count=channel_count,
                             nominal_srate=nominal_srate, channel_format=channel_format, source_id=source_id)

            # Add channel labels
            channels = info.desc().append_child("channels")
            for idx, channel in enumerate(self.exg_channels):
                channels.append_child("channel").append_child_value("label", f"CH{channel+1}")

            self.outlet = StreamOutlet(info)
            logging.info(f"LSL Stream '{stream_name}' initialized with {channel_count} channels.")
        except Exception as e:
            logging.error(f"Failed to initialize LSL stream: {e}")
            self.release_board()
            sys.exit(1)

    def _init_timeseries(self):
        """Initializes the time series plots for each channel."""
        self.plots = []
        self.curves = []
        for i in range(len(self.exg_channels)):
            p = self.win.addPlot(row=i, col=0)
            p.showAxis('left', False)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            if i == 0:
                p.setTitle('TimeSeries Plot')
            self.plots.append(p)
            curve = p.plot(pen=pg.mkPen(color=(255, 0, 0), width=1))
            self.curves.append(curve)
        logging.info("Time series plots initialized.")

    def update(self):
        """Fetches new data from the board, updates plots, and enqueues data for LSL streaming."""
        try:
            # Fetch the latest 'num_points' samples
            new_data = self.board_shim.get_current_board_data(self.num_points)
            if new_data is None or new_data.size == 0:
                logging.debug("No new data received.")
                return

            # Extract EXG channels
            exg_data = new_data[self.exg_channels, :]  # shape: (channels, samples)

            # Apply filters and update plots
            for count, channel_data in enumerate(exg_data):
                # Ensure C-contiguous memory layout
                channel_data = channel_data.copy()

                # Detrend
                DataFilter.detrend(channel_data, DetrendOperations.CONSTANT.value)

                # Bandpass filter
                DataFilter.perform_bandpass(channel_data, self.sampling_rate, 4.0, 250, 2, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
                
                # Bandstop filter to remove 58-62 Hz noise
                DataFilter.perform_bandstop(channel_data, self.sampling_rate, 48.0, 52.0, 2,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
                DataFilter.perform_bandstop(channel_data, self.sampling_rate, 58.0, 62.0, 2,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)

                # Notch filter to remove powerline noise
                DataFilter.remove_environmental_noise(channel_data, self.sampling_rate, NoiseTypes.FIFTY_AND_SIXTY)

                # Update plot
                self.curves[count].setData(channel_data.tolist())

            # Enqueue data for LSL streaming
            num_samples = exg_data.shape[1]
            for sample_idx in range(num_samples):
                sample = exg_data[:, sample_idx].tolist()
                self.data_queue.put(sample)

        except BrainFlowError as e:
            logging.error(f"BrainFlowError in update function: {e}")
            self.release_board()
            sys.exit(1)
        except Exception as e:
            logging.error(f"Unexpected error in update function: {e}")
            self.release_board()
            sys.exit(1)

    def on_close(self, event):
        """Handle the window close event to release board resources."""
        logging.info("Window closed by user.")
        self.release_board()
        event.accept()

    def release_board(self):
        """Releases board resources and stops LSL streaming."""
        try:
            if self.board_shim.is_prepared():
                logging.info('Stopping and releasing board session.')
                self.board_shim.stop_stream()
                self.board_shim.release_session()
        except Exception as e:
            logging.error(f"Error releasing board: {e}")

        # Signal the LSLStreamer thread to stop and wait for it to finish
        self.stop_event.set()
        self.lsl_thread.join()
        logging.info("LSLStreamer thread has been stopped.")

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    BoardShim.enable_dev_board_logger()

    try:
        serial_port = find_serial_port()
        board_id = BoardIds.CYTON_BOARD
        emg = True

        graph = Graph(board_id=board_id, emg=emg, serial_port=serial_port)
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

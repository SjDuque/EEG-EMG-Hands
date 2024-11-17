import argparse
import logging
import pyqtgraph as pg
from brainflow import NoiseTypes
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from PyQt5 import QtWidgets, QtCore 

import serial.tools.list_ports

def find_serial_port():
    """
    Automatically detect the serial port for the OpenBCI board.
    Returns the port name as a string, or raises an exception if not found.
    """
    ports = serial.tools.list_ports.comports()
    for port in ports:
        # Check for specific keywords or VID/PID
        if 'OpenBCI' in port.description or 'usbserial' in port.device:
            return port.device
    raise RuntimeError("OpenBCI board not found. Ensure it is connected.")


class Graph:
    def __init__(self, board_id=BoardIds.SYNTHETIC_BOARD, emg=False, serial_port=''):
        self.params = BrainFlowInputParams()
        self.params.serial_port = serial_port
        self.board_id = board_id
        self.emg = emg

        # Initialize the board
        self.board_shim = BoardShim(self.board_id, self.params)
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        
        # Set up the board
        self._setup_board()

        self.update_speed_ms = 50
        self.window_size = 2  # 2 seconds of data
        self.num_points = self.window_size * self.sampling_rate

        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(title='BrainFlow Plot')
        self.win.resize(800, 600)
        self.win.setWindowTitle('BrainFlow Plot')
        self.win.show()

        self._init_timeseries()

        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(self.update_speed_ms)
        self.app.exec_()

    def _setup_board(self):
        """Prepares the board session and sets SRB2 configuration if needed."""
        try:
            self.board_shim.prepare_session()
            config_srb2_list = []
            if self.emg and self.board_id in (BoardIds.CYTON_BOARD, BoardIds.CYTON_DAISY_BOARD):
                for channel in self.exg_channels:
                    config_srb2_list.append(f"x{channel}060100")
            # Set SRB2 configuration
            config_srb2 = ''.join(config_srb2_list)
            self.board_shim.config_board(config_srb2)
            self.board_shim.start_stream(450000)
        except BaseException as e:
            logging.warning("Exception during board setup", exc_info=True)
            raise e

    def _init_timeseries(self):
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
            curve = p.plot()
            self.curves.append(curve)

    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points)
        for count, channel in enumerate(self.exg_channels):
            # Detrend
            DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
            
            # Bandpass filter
            DataFilter.perform_bandpass(data[channel], self.sampling_rate, 4.0, 250, 2, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
            
            # Notch filter
            DataFilter.remove_environmental_noise(data[channel], self.sampling_rate, NoiseTypes.FIFTY_AND_SIXTY)
            
            # Update Plot
            self.curves[count].setData(data[channel].tolist())
            
        self.app.processEvents()

    def release_board(self):
        """Releases board resources."""
        if self.board_shim.is_prepared():
            logging.info('Releasing session')
            self.board_shim.release_session()


def main():
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    
    serial_port = find_serial_port()
    board_id = BoardIds.CYTON_BOARD
    emg = True

    try:
        graph = Graph(board_id=board_id, emg=emg, serial_port=serial_port)
    except BaseException:
        logging.warning('Exception', exc_info=True)
    finally:
        logging.info('End')


if __name__ == '__main__':
    main()

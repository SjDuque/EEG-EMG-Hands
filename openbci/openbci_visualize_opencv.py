import logging
import cv2
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from brainflow.exit_codes import BrainFlowError

logging.basicConfig(level=logging.INFO)

class Graph:
    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate

        self.window_name = "BrainFlow Plot"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.window_height = 600
        self.window_width = 800

        self._init_timeseries()

        while True:
            self.update()
            if cv2.waitKey(self.update_speed_ms) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def _init_timeseries(self):
        self.plot_data = np.zeros((len(self.exg_channels), self.num_points))

    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points)
        if data is None or data.shape[1] == 0:
            logging.warning('Received empty data array.')
            return

        actual_points = data.shape[1]
        for count, channel in enumerate(self.exg_channels):
            channel_data = data[channel][:actual_points]
            if len(channel_data) == 0:
                logging.warning(f'No data for channel {channel}')
                continue
            
            try:
                # Process data
                DataFilter.detrend(channel_data, DetrendOperations.CONSTANT.value)
                DataFilter.perform_bandpass(channel_data, self.sampling_rate, 3.0, 45.0, 2,
                                            FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
                DataFilter.perform_bandstop(channel_data, self.sampling_rate, 48.0, 52.0, 2,
                                            FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
                DataFilter.perform_bandstop(channel_data, self.sampling_rate, 58.0, 62.0, 2,
                                            FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
                self.plot_data[count, :actual_points] = channel_data
            except BrainFlowError as e:
                logging.error(f'Error processing data for channel {channel}: {e}')
                continue

        self.plot(actual_points)

    def plot(self, actual_points):
        img = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
        for i in range(len(self.exg_channels)):
            y = self.plot_data[i, :actual_points] - np.min(self.plot_data[i, :actual_points])
            if np.max(y) != 0:
                y = y / np.max(y) * (self.window_height // len(self.exg_channels))
            y = y.astype(np.int32)
            x = np.linspace(0, self.window_width, actual_points, dtype=np.int32)
            for j in range(len(y) - 1):
                cv2.line(img, (x[j], i * self.window_height // len(self.exg_channels) + y[j]),
                         (x[j + 1], i * self.window_height // len(self.exg_channels) + y[j + 1]), (255, 255, 255), 1)
        cv2.imshow(self.window_name, img)


def main():
    # Setup BrainFlow parameters
    params = BrainFlowInputParams()
    params.serial_port = '/dev/cu.usbserial-D30DLPKS'
    board_id = BoardIds.CYTON_BOARD.value

    # Initialize the board
    board_shim = BoardShim(board_id, params)
    try:
        board_shim.prepare_session()
        board_shim.start_stream(450000)
        Graph(board_shim)
    except BaseException as e:
        logging.warning(f'Exception: {e}', exc_info=True)
    finally:
        logging.info('End')
        if board_shim.is_prepared():
            logging.info('Releasing session')
            board_shim.release_session()


if __name__ == '__main__':
    main()

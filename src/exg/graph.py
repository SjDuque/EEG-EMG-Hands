# graph.py
import logging
import sys
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QShortcut
from PyQt5.QtGui import QKeySequence

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

class BaseGraph:
    """
    Base class for graphing functionalities.
    Initializes the PyQt application and sets up the plotting window.
    """
    def __init__(self, title='Graph Plotter', window_size=5, sampling_rate=250, num_channels=1):
        self.window_size = window_size  # seconds
        self.sampling_rate = sampling_rate
        self.buffer_size = int(self.window_size * self.sampling_rate)
        self.num_channels = num_channels

        # Initialize PyQt application and window
        self.app = QtWidgets.QApplication(sys.argv)
        self.win = CustomGraphicsLayoutWidget(title=title)
        self.win.resize(800, 600)
        self.win.setWindowTitle(title)
        self.win.show()

        # Ensure that resources are released when window is closed
        self.win.closeEvent = self.on_close

        # Add shortcut for ESC key to close the window
        shortcut = QShortcut(QKeySequence("Esc"), self.win)
        shortcut.activated.connect(self.win.close)

        # Initialize plots
        self.plots = []
        self.curves = []
        self._init_plots()

    def _init_plots(self):
        """
        Initializes the time series plots for each channel.
        """
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'w']  # Define more colors as needed
        for i in range(self.num_channels):
            p = self.win.addPlot(row=i, col=0)
            p.showAxis('left', True)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            if i == 0:
                p.setTitle('Time Series Data')

            # Set fixed y-axis limits for the plot (adjust as per your data)
            p.setYRange(-250, 250)

            # Label each channel
            p.getAxis('left').setLabel(f'Ch {i+1}')

            self.plots.append(p)
            color = colors[i % len(colors)]
            curve = p.plot(pen=pg.mkPen(color=color, width=2))
            self.curves.append(curve)

    def update_plots(self, data_buffers):
        """
        Updates the plots with new data.
        
        :param data_buffers: A 2D NumPy array of shape (num_channels, buffer_size)
        """
        for i in range(self.num_channels):
            self.curves[i].setData(data_buffers[i])

    def on_close(self, event):
        """
        Handle the window close event to perform any necessary cleanup.
        """
        logging.info("Window closed by user.")
        self.cleanup()
        event.accept()

    def cleanup(self):
        """
        Cleanup resources. To be overridden by subclasses if needed.
        """
        pass

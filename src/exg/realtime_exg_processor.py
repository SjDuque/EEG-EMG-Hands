import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi

## TODO:
# 0. Integrate filter design into the RealtimeEXGProcessor class.
# 1. Bandpass filter design 
# 2. Notch filter (or bandstop filter) design
# 3. SOS filter design
# 4. Simplify the attributes and methods of the RealtimeEXGProcessor class. e.g., remove self.sums and self.squared_sums.
# 5. Add a method to reset the processor.
# 6. Add a method to compute the metrics for a single window size.
# 7. Add a method to compute the metrics for a single point of data.
# 8. Dimensionality of the result: (num_samples, num_windows, num_channels).
# 9. Take into account when the total number of samples is less than the window size. (Or just set default values to 0)
# 10. Also add filtered data to the results. (bandpass and notch)
# 11. Simplify index management.
# 12. Modularize the code so there is no unnecessary overhead (for example, only needing bandpass filter).  

# Define helper functions for filter design and real-time IIR filtering
def design_bandpass_filter(low_cut, high_cut, fs, order=4):
    """
    Design a Butterworth bandpass filter.

    Args:
        low_cut (float): Low cutoff frequency in Hz.
        high_cut (float): High cutoff frequency in Hz.
        fs (float): Sampling frequency in Hz.
        order (int): Filter order (default: 4).

    Returns:
        b, a (ndarray, ndarray): Numerator (b) and denominator (a) coefficients of the filter.
    """
    nyquist = 0.5 * fs
    b, a = butter(order, [low_cut / nyquist, high_cut / nyquist],
                 btype='band', analog=False)
    return b, a

def real_time_bandpass_filter(data, b, a, zi):
    """
    Apply bandpass filtering to multi-channel data in real-time using lfilter.

    Args:
        data (ndarray): New data batch (num_channels, batch_size).
        b (ndarray): Numerator coefficients of the filter.
        a (ndarray): Denominator coefficients of the filter.
        zi (ndarray): Initial filter states (num_channels, max(len(a), len(b)) - 1).

    Returns:
        filtered_data (ndarray): Filtered data.
        updated_zi (ndarray): Updated filter states.
    """
    num_channels, batch_size = data.shape
    filtered_data = np.empty_like(data)
    updated_zi = np.empty_like(zi)
    
    for ch in range(num_channels):
        # Apply lfilter to each channel with its own state
        filtered_data[ch], updated_zi[ch] = lfilter(b, a, data[ch], zi=zi[ch])
    
    return filtered_data, updated_zi

class RealtimeEXGProcessor:
    def __init__(self, methods:set[str], window_intervals:list[float], num_channels:int, fs:float, 
                 low_pass_cutoff:float = None, high_pass_cutoff:float = None, 
                 filter_order:int = 4):
        """
        Initialize the rolling processor with a bandpass filter and multiple window sizes.

        Args:
            methods (set): Set of processing methods to compute.
            window_intervals (list): List of window sizes in seconds.
            num_channels (int): Number of signal channels.
            fs (float): Sampling frequency in Hz.
            low_pass_cutoff (float, optional): Low-pass filter cutoff frequency in Hz.
            high_pass_cutoff (float, optional): High-pass filter cutoff frequency in Hz.
            filter_order (int, optional): Filter order (default: 4).
        """
        window_sizes = [int(interval * fs) for interval in window_intervals]  # Convert seconds to number of samples
        
        self.window_sizes = sorted(window_sizes)
        self.num_channels = num_channels
        self.num_windows = len(self.window_sizes)
        self.max_window_size = self.window_sizes[-1]
        
        # Initialize rolling sums and squared sums for each window size
        self.sums = {w: np.zeros((num_channels,)) for w in self.window_sizes}
        self.squared_sums = {w: np.zeros((num_channels,)) for w in self.window_sizes}

        # Initialize rolling buffers
        self.prev_vals = np.zeros((num_channels, self.max_window_size))
        self.indices = np.zeros(num_channels, dtype=int)  # Current index for each channel

        # Design bandpass filter if both low and high cutoffs are provided
        self.b = None
        self.a = None
        self.zi = None

        # Initialize processing methods
        self.methods = methods

        if low_pass_cutoff and high_pass_cutoff:
            self.b, self.a = design_bandpass_filter(high_pass_cutoff, low_pass_cutoff, fs, order=filter_order)
            # Initialize filter states for each channel
            # lfilter_zi returns the steady-state zi for step response, which can be scaled by the initial condition
            self.zi = lfilter_zi(self.b, self.a)
            self.zi = np.tile(self.zi, (num_channels, 1)) * 0  # Initialize to zero or some initial condition

    def _apply_bandpass_filter(self, data):
        """
        Apply the bandpass filter to the data.

        Args:
            data (ndarray): Input data (num_channels, batch_size).

        Returns:
            filtered_data (ndarray): Filtered data.
        """
        if self.b is not None and self.a is not None:
            filtered_data, self.zi = real_time_bandpass_filter(data, self.b, self.a, self.zi)
            return filtered_data
        return data

    def _update_buffer(self, new_vals):
        """
        Update the rolling buffer with new values for all channels.

        Args:
            new_vals (ndarray): Array of new values for all channels. Shape: (num_channels,).

        Returns:
            oldest_vals (ndarray): Values being replaced in the buffer. Shape: (num_channels,).
        """
        idx = self.indices
        oldest_vals = self.prev_vals[np.arange(self.num_channels), idx]
        self.prev_vals[np.arange(self.num_channels), idx] = new_vals
        return oldest_vals

    def update(self, new_vals):
        """
        Update the rolling processor with new values for all channels and compute selected methods.

        Args:
            new_vals (ndarray): New signal values for all channels. Shape: (num_channels, batch_size).

        Returns:
            dict: Metrics for each method and window size.
                  Each metric is a NumPy array of shape (num_channels, batch_size).
        """

            # Apply filters
        filtered_vals = self._apply_bandpass_filter(new_vals)
        filtered_vals = np.abs(filtered_vals)  # Rectify the signal

        batch_size = filtered_vals.shape[1]
        results = {method: {w: np.empty((self.num_channels, batch_size)) for w in self.window_sizes} for method in self.methods}

        for b in range(batch_size):
            # Extract new values for this timestep
            batch = filtered_vals[:, b]

            # Update the rolling buffer and get oldest values
            oldest_vals = self._update_buffer(batch)

            # Update rolling sums and squared sums
            for w in self.window_sizes:
                self.sums[w] += batch - oldest_vals
                self.squared_sums[w] += batch**2 - oldest_vals**2
                
                mean = self.sums[w] / w
                mean_square = self.squared_sums[w] / w
                
                # Consider dividing by the window size to compute the mean and variance

                if "mean" in self.methods:
                    # Compute the mean value in the current window
                    results["mean"][w][:, b] = mean

                if "mean_squared" in self.methods:
                    # Compute the mean squared value in the current window
                    results["mean_squared"][w][:, b] = mean_square

                if "root_mean_squared" in self.methods:
                    # Compute the root mean squared value in the current window
                    results["root_mean_squared"][w][:, b] = np.sqrt(mean_square)

                if "variance" in self.methods:
                    # Compute the variance in the current window
                    results["variance"][w][:, b] = mean_square - mean**2

                if "peak" in self.methods:
                    # Compute the peak (max) in the current window
                    idxs = (self.indices[:, None] - np.arange(w)) % self.max_window_size
                    windowed_vals = self.prev_vals[np.arange(self.num_channels)[:, None], idxs]
                    results["peak"][w][:, b] = np.max(windowed_vals, axis=1)

            # Update the rolling index for each channel
            self.indices = (self.indices + 1) % self.max_window_size

        return results

def main():
    # Define parameters
    window_intervals = [0.020, 0.050, 0.100]  # Rolling window sizes in seconds
    num_channels = 8  # Number of EMG/EEG channels
    fs = 250.0  # Sampling frequency in Hz
    filter_order = 4  # IIR filter order
    low_pass_cutoff = fs / 2  # Low-pass filter cutoff frequency in Hz
    high_pass_cutoff = 10.0  # High-pass filter cutoff frequency in Hz

    methods = set(["mean", "mean_squared", "root_mean_squared", "variance", "peak"])

    # Initialize the rolling processor
    processor = RealtimeEXGProcessor(
        methods=methods,
        window_intervals=window_intervals,
        num_channels=num_channels,
        fs=fs,
        low_pass_cutoff=low_pass_cutoff,
        high_pass_cutoff=high_pass_cutoff,
        filter_order=filter_order
    )

    print("Data acquisition and processing completed.")

if __name__ == "__main__":
    main()

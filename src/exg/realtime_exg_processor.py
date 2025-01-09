import numpy as np
import pandas as pd

## TODO:
# 0. Integrate filter design into the RealtimeEXGProcessor class.
# 1. Bandpass filter design
# 2. Notch filter (or bandstop filter) design
# 3. SOS filter design
# 4. Simplify the attributes and methods of the RealtimeEXGProcessor class
# 5. Add a method to reset the processor.
# 6. Add a method to compute the metrics for a single window size.
# 7. Add a method to compute the metrics for a single point of data.
# 8. Dimensionality of the result: (num_samples, num_windows, num_channels).
# 9. Take into account when the total number of samples is less than the window size. (Or just set default values to 0)
# 10. Also add filtered data to the results. (bandpass and notch)
# 11. Simplify index management.
# 12. Modularize the code so there is no unnecessary overhead (for example, only needing bandpass filter).

class RealtimeEXGProcessor:
    def __init__(self, num_channels:int, fs:float, window_intervals:list[float]=None, window_sizes:list[int]=None,
                 methods:list[str] = ['mean', 'mean_square', 'root_mean_square', 'variance', 'standard_deviation']
                 ):
        """
        Initialize the rolling processor with multiple window sizes.

        Args:
            num_channels (int): Number of signal channels.
            fs (float): Sampling frequency in Hz.
            window_intervals (list): List of window sizes in seconds. (Choose either window_intervals or window_sizes.)
            window_sizes (list): List of window sizes in number of samples. (Choose either window_intervals or window_sizes.)
            methods (list): List of processing methods to compute. Default: ALL
        """
        # Initialize processing methods
        self.methods = set(methods)

        # Modify the methods to include the requisite methods
        if "standard_deviation" in self.methods:
            self.methods.add("variance")

        if "variance" in self.methods:
            self.methods.add("mean_square")

        if "root_mean_square" in self.methods:
            self.methods.add("mean_square")

        if "mean_square" in self.methods:
            self.methods.add("mean")
        
        # Initialize window sizes
        if window_intervals is None and window_sizes is None:
            raise ValueError("Either window_intervals or window_sizes must be provided.")
        elif window_intervals is not None and window_sizes is not None:
            raise ValueError("Only one of window_intervals or window_sizes can be provided.")
        elif window_intervals is not None:
            self.window_sizes = sorted([int(round(interval * fs)) for interval in window_intervals])
        elif window_sizes is not None:
            self.window_sizes = sorted(window_sizes)

        # Initialize attributes
        self.num_channels    = num_channels
        self.num_windows     = len(self.window_sizes)
        self.max_window_size = self.window_sizes[-1]

        # Initialize rolling buffers
        self.idx            = 0
        self.prev_vals      = np.zeros((self.max_window_size, num_channels), dtype=np.float64)
        if "mean" in self.methods:
            self.sum        = np.zeros((self.num_windows, num_channels), dtype=np.float64)
        if "mean_square" in self.methods:
            self.sum_square = np.zeros((self.num_windows, num_channels), dtype=np.float64)

    def _push_sample(self, new_sample:np.ndarray):
        """
        Update the rolling buffer with new values for all channels.

        Args:
            new_vals (ndarray): Array of new values for all channels. Shape: (num_channels,).
        """
        if new_sample.shape != self.prev_vals[self.idx].shape:
            raise ValueError(f"Shape of new sample ({new_sample.shape}) does not match the expected shape ({self.prev_vals[self.idx].shape}).")

        # Update the rolling buffer
        self.prev_vals[self.idx, :] = new_sample
        self.idx                    = (self.idx + 1) % self.max_window_size

    def process(self, new_vals:np.ndarray | list[list[float]]) -> dict[str, dict[int, np.ndarray]]:
        """
        Update the rolling processor with new values for all channels and compute selected methods.

        Args:
            new_vals (ndarray): New signal values for all channels. Shape: (num_samples, num_channels).

        Returns:
            dict: Metrics for each method
                  Each metric is a NumPy array of shape (num_samples, num_windows, num_channels).
        """
        # Rectify the signal
        new_vals = np.abs(new_vals)

        # Get the number of samples
        num_samples, num_channels = new_vals.shape
        if num_channels != self.num_channels:
            raise ValueError(f"Number of channels ({num_channels}) does not match the expected value ({self.num_channels}).")

        # Initialize results for each method in self.methods and window size
        results = {
            method: np.zeros((num_samples, self.num_windows, num_channels), dtype=np.float64)
            for method in self.methods
        }
        for s in range(num_samples):
            new_sample = new_vals[s]
            # Update rolling sums and squared sums
            for w in range(self.num_windows):
                window_size = self.window_sizes[w]
                old_sample = self.prev_vals[self.idx - window_size]
                
                mean = None
                mean_square = None
                variance = None

                if "mean" in self.methods:
                    # Compute the mean value in the current window
                    self.sum[w] += new_sample - old_sample
                    self.sum[w] = np.clip(self.sum[w], 0, None)
                    mean = self.sum[w] / window_size
                    results["mean"][s, w, :] = mean

                if "mean_square" in self.methods:
                    # Compute the mean squared value in the current window
                    self.sum_square[w] += new_sample**2 - old_sample**2
                    self.sum_square[w] = np.clip(self.sum_square[w], 0, None)
                    mean_square = self.sum_square[w] / window_size
                    results["mean_square"][s, w, :] = mean_square

                if "root_mean_square" in self.methods:
                    # Compute the root mean squared value in the current window
                    results["root_mean_square"][s, w, :] = np.sqrt(mean_square)

                if "variance" in self.methods:
                    # Compute the variance in the current window
                    variance = mean_square - mean**2
                    variance = np.clip(variance, 0, None)
                    results["variance"][s, w, :] = variance

                if "standard_deviation" in self.methods:
                    # Compute the standard deviation in the current window
                    results["standard_deviation"][s, w, :] = np.sqrt(variance)

            # Update the rolling index for each channel
            self._push_sample(new_sample)
            
        return results
    
    def results_to_df(self, results: dict[str, dict[int, np.ndarray]]) -> pd.DataFrame:
        """
        Convert the results to a Pandas DataFrame.

        Args:
            results (dict): Dictionary of results for each method.
                            Each result is a NumPy array of shape (num_samples, num_windows, num_channels).

        Returns:
            DataFrame: DataFrame of the results.
        """
        data = {}
        for method, result in results.items():
            for w in range(self.num_windows):
                window_size = self.window_sizes[w]
                columns = [f"ch_{c+1}_{method}_{window_size}" for c in range(self.num_channels)]
                data.update({
                    col: result[:, w, c] for c, col in enumerate(columns)
                })
        return pd.DataFrame(data)
    
    def reset(self):
        """
        Reset the rolling processor.
        """
        self.idx = 0
        self.prev_vals.fill(0)
        if "mean" in self.methods:
            self.sum.fill(0)
        if "mean_square" in self.methods:
            self.sum_square.fill(0)
        
        print("Processor reset.")

def main():
    # Define parameters
    window_intervals = [0.020, 0.050, 0.100]  # Rolling window sizes in seconds
    num_channels = 8  # Number of EMG/EEG channels
    fs = 250.0  # Sampling frequency in Hz

    methods = ["mean", "mean_square", "root_mean_square", "variance"]

    # Initialize the rolling processor
    processor = RealtimeEXGProcessor(
        methods=methods,
        window_intervals=window_intervals,
        num_channels=num_channels,
        fs=fs
    )

    print("Data acquisition and processing completed.")

if __name__ == "__main__":
    main()

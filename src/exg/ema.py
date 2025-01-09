import numpy as np
import pandas as pd

class EMA:
    """
    _summary_: This class implements a exponential moving average (EMA) processor with multiple window sizes.
    """
    def __init__(self, num_channels:int, fs:float, window_intervals:list[float]=None, window_sizes:list[int]=None,
                 methods:list[str] = ['mean', 'mean_square', 'root_mean_square', 'variance', 'standard_deviation']
                 ):
        """
        Initialize the exponential moving average

        Args:
            num_channels (int): Number of signal channels.
            fs (float): Sampling frequency in Hz.
            window_intervals (list): List of window sizes in seconds. (Choose either window_intervals or window_sizes.)
            window_sizes (list): List of window sizes in number of samples. (Choose either window_intervals or window_sizes.)
            methods (list): List of processing methods to compute. Default: ALL
        """
        # Initialize window sizes
        if window_intervals is None and window_sizes is None:
            raise ValueError("Either window_intervals or window_sizes must be provided.")
        elif window_intervals is not None and window_sizes is not None:
            raise ValueError("Only one of window_intervals or window_sizes can be provided.")
        elif window_intervals is not None:
            self.window_sizes = sorted([interval * fs for interval in window_intervals])
        elif window_sizes is not None:
            self.window_sizes = sorted(window_sizes)
            
        self.alphas = [self._window_to_alpha(window_size, fs) for window_size in self.window_sizes]
        
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

        # Initialize attributes
        self.num_channels    = num_channels
        self.num_windows     = len(self.window_sizes)

        # Initialize ema
        if "mean" in self.methods:
            self.ema        = np.zeros((self.num_windows, num_channels), dtype=np.float64)
        if "mean_square" in self.methods:
            self.ema_square = np.zeros((self.num_windows, num_channels), dtype=np.float64)
    
    @staticmethod
    def _window_to_alpha(window_size:int, fs:float) -> float:
        """
        Convert a window size to the smoothing factor alpha.

        Args:
            window_size (int): Window size in number of samples.

        Returns:
            float: Smoothing factor alpha.
        """
        if window_size < 1:
            raise ValueError("Window size must be greater than 0.")
        return 2 / (window_size + 1)

    def process(self, new_vals:np.ndarray | list[list[float]]) -> dict[str, dict[int, np.ndarray]]:
        """
        Update the rolling processor with new values for all channels and compute selected methods.

        Args:
            new_vals (ndarray): New signal values for all channels. Shape: (num_samples, num_channels).

        Returns:
            dict: Metrics for each method
                  Each metric is a NumPy array of shape (num_windows, num_samples, num_channels).
        """
        # Rectify the signal
        new_vals = np.abs(new_vals)

        # Get the number of samples
        num_samples, num_channels = new_vals.shape
        if num_channels != self.num_channels:
            raise ValueError(f"Number of channels ({num_channels}) does not match the expected value ({self.num_channels}).")

        # Initialize results for each method in self.methods and window size
        results = {
            method: np.zeros((self.num_windows, num_samples, num_channels), dtype=np.float64)
            for method in self.methods
        }
        for s in range(num_samples):
            new_sample = new_vals[s]
            # Update rolling sums and squared sums
            for w in range(self.num_windows):
                mean = None
                mean_square = None
                variance = None

                if "mean" in self.methods:
                    # Compute the mean value in the current window
                    self.ema[w] *= 1 - self.alphas[w]
                    self.ema[w] += new_sample * self.alphas[w]
                    mean = self.ema[w]
                    results["mean"][w, s, :] = mean

                if "mean_square" in self.methods:
                    # Compute the mean squared value in the current window
                    self.ema_square[w] *= 1 - self.alphas[w]
                    self.ema_square[w] += (new_sample * new_sample) * self.alphas[w]
                    mean_square = self.ema_square[w]
                    results["mean_square"][w, s, :] = mean_square

                if "root_mean_square" in self.methods:
                    # Compute the root mean squared value in the current window
                    results["root_mean_square"][w, s, :] = np.sqrt(mean_square)

                if "variance" in self.methods:
                    # Compute the variance in the current window
                    variance = mean_square - mean**2
                    variance = np.clip(variance, 0, None)
                    results["variance"][w, s, :] = variance

                if "standard_deviation" in self.methods:
                    # Compute the standard deviation in the current window
                    results["standard_deviation"][w, s, :] = np.sqrt(variance)
            
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
                window_size = int(round(self.window_sizes[w]))
                columns = [f"ch_{c+1}_{method}_{window_size}" for c in range(self.num_channels)]
                data.update({
                    col: result[:, w, c] for c, col in enumerate(columns)
                })
        return pd.DataFrame(data)
    
    def reset(self):
        """
        Reset the rolling processor.
        """
        if "mean" in self.methods:
            self.ema.fill(0)
        if "mean_square" in self.methods:
            self.ema_square.fill(0)
        
        print("Processor reset.")

def main():
    # Define parameters
    window_intervals = [0.020, 0.050, 0.100]  # Rolling window sizes in seconds
    num_channels = 8  # Number of EMG/EEG channels
    fs = 250.0  # Sampling frequency in Hz

    methods = ["mean", "mean_square", "root_mean_square", "variance", "standard_deviation"]

    # Initialize the rolling processor
    processor = EMA(
        methods=methods,
        window_intervals=window_intervals,
        num_channels=num_channels,
        fs=fs
    )
    
    # Generate random data
    num_samples = 1000
    data = np.random.randn(num_samples, num_channels)
    # Process the data
    results = processor.process(data)
    # Mean values in the first window
    print('Mean:', results["mean"][0].mean())
    # Standard deviation values in the second window
    print('STD:', results["standard_deviation"][0].mean())

    print("Data acquisition and processing completed.")

if __name__ == "__main__":
    main()

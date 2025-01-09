import numpy as np
import pandas as pd
from scipy.signal import lfilter

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
        self.num_channels = num_channels
        self.num_windows  = len(self.window_sizes)

        # Initialize ema
        if "mean" in self.methods:
            self.zf_mean        = np.zeros((self.num_windows, 1, num_channels), dtype=np.float64)
        if "mean_square" in self.methods:
            self.zf_mean_square = np.zeros((self.num_windows, 1, num_channels), dtype=np.float64)
    
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

    def process(self, new_vals: np.ndarray) -> dict[str, np.ndarray]:
        """
        Vectorized processing of new values for all channels.
        new_vals shape: (num_samples, num_channels)

        Returns:
            dict: Dict of metric -> ndarray of shape (num_windows, num_samples, num_channels)
        """
        # Rectify
        new_vals = np.abs(new_vals)
        num_samples, num_channels = new_vals.shape
        
        if num_channels != self.num_channels:
            raise ValueError(
                f"Number of channels ({num_channels}) does not match expected ({self.num_channels})."
            )

        # Prepare outputs
        results = {
            method: np.zeros((self.num_windows, num_samples, num_channels), dtype=np.float64)
            for method in self.methods
        }

        # Precompute squares
        if "mean_square" in self.methods:
            new_vals_sq = new_vals * new_vals

        for w in range(self.num_windows):
            alpha = self.alphas[w]
            
            # Filter coefficients for an EMA
            # y[n] = alpha*x[n] + (1-alpha)*y[n-1]
            b = [alpha]     
            a = [1, alpha - 1]

            # ---------------
            # 1) Mean 
            # ---------------
            if "mean" in self.methods:
                # run lfilter for mean, passing in the saved state
                y_mean, zf_m = lfilter(
                    b, a, new_vals, axis=0, zi=self.zf_mean[w]
                )
                # shape of y_mean => (num_samples, num_channels)
                results["mean"][w] = y_mean
                # store updated final state
                self.zf_mean[w] = zf_m

            # ---------------
            # 2) Mean square
            # ---------------
            if "mean_square" in self.methods:
                y_msq, zf_msq = lfilter(
                    b, a, new_vals_sq, axis=0, zi=self.zf_mean_square[w]
                )
                results["mean_square"][w] = y_msq
                self.zf_mean_square[w] = zf_msq

            # ---------------
            # 3) Derived metrics: RMS, variance, std
            #    We can compute them from y_mean and y_msq
            #    If for some reason "mean" or "mean_square"
            #    wasn't in self.methods, you can do a second 
            #    call to lfilter. But typically we store them above.
            # ---------------
            if "variance" in self.methods or "standard_deviation" in self.methods or "root_mean_square" in self.methods:
                # Safely reference them:
                if "mean" in self.methods:
                    y_mean_ema = results["mean"][w]
                else:
                    # if "mean" was omitted, you'd do another lfilter call 
                    # or handle differently
                    y_mean_ema, _ = lfilter(b, a, new_vals, axis=0, zi=self.zf_mean[w])

                if "mean_square" in self.methods:
                    y_msq_ema = results["mean_square"][w]
                else:
                    y_msq_ema, _ = lfilter(b, a, new_vals_sq, axis=0, zi=self.zf_mean_square[w])

                # Variance = E[X^2] - (E[X])^2
                variance = y_msq_ema - (y_mean_ema ** 2)
                variance = np.clip(variance, 0.0, None)

                if "variance" in self.methods:
                    results["variance"][w] = variance

                if "standard_deviation" in self.methods:
                    results["standard_deviation"][w] = np.sqrt(variance)

                if "root_mean_square" in self.methods:
                    results["root_mean_square"][w] = np.sqrt(y_msq_ema)

        return results

    def reset(self):
        """ Reset the filter states to 0. """
        self.zf_mean.fill(0.0)
        self.zf_mean_square.fill(0.0)
        print("Processor state reset.")
    
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
    num_samples = 100000
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

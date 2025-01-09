import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, tf2sos

class IIR:
    """
    _summary_: This class implements Infinite Impulse Response (IIR) filtering for real-time processing of multi-channel data.
    """
    def __init__(self, num_channels:int, fs:float, lowpass_fs:float, highpass_fs:float, 
                 notch_fs_list:list[float], filter_order:int=4, quality_factor:float=50):
        pass

    @staticmethod
    def design_bandpass_filter(fs:float, lowpass_fs:float, highpass_fs:float, order:int=4, eps=1e-8):
        """
        Design a Butterworth bandpass filter using Second-Order Sections (SOS).

        Args:
            lowpass_fs (float): Low cutoff frequency in Hz.
            highpass_fs (float): High cutoff frequency in Hz.
            fs (float): Sampling frequency in Hz.
            order (int): Filter order (default: 4).

        Returns:
            sos (ndarray): Second-Order Sections representation of the filter.
        """
        nyquist = 0.5 * fs
        Wn = [highpass_fs / nyquist, lowpass_fs / nyquist]

        # Ensure cutoff frequencies are within valid range
        Wn[0] = max(Wn[0], eps)  # Ensure low cutoff is at least 0.00...1% of Nyquist
        Wn[1] = min(Wn[1], 1-eps)  # Ensure high cutoff is at most 99.99...9% of Nyquist
        if Wn[0] >= Wn[1]:
            raise ValueError("High cutoff frequency must be greater than low cutoff frequency.")
        sos = butter(order, Wn=Wn, btype='band', analog=False, output='sos')
        return sos

    @staticmethod
    def design_notch_filter(fs:float, notch_fs:float, quality_factor:float=50):
        """
        Design a notch filter using Second-Order Sections (SOS).

        Args:
            fs (float): Sampling frequency in Hz.
            notch_fs (float): Frequency to notch out in Hz.
            quality_factor (float): Quality factor for the notch filter.

        Returns:
            sos (ndarray): Second-Order Sections representation of the notch filter.
        """
        b, a = iirnotch(notch_fs, quality_factor, fs=fs)
        sos = tf2sos(b, a)
        return sos

    @staticmethod
    def apply_bandpass_inplace(data:np.ndarray, sos:np.ndarray, zi:np.ndarray):
        """
        Apply bandpass filtering to multi-channel data in real-time using sosfilt.

        Args:
            data (ndarray): New data batch (num_samples, num_channels).
            sos (ndarray): Second-Order Sections representation of the filter.
            zi (ndarray): Initial filter states (num_channels, n_sections, 2).

        Returns:
            filtered_data (ndarray): Filtered data.
            updated_zi (ndarray): Updated filter states.
        """
        num_channels = data.shape[1]

        for ch in range(num_channels):
            # Apply sosfilt to each channel with its own state
            data[:, ch], zi[ch] = sosfilt(sos, data[ch], zi=zi[ch])
    
    @staticmethod
    def apply_notch_inplace(data:np.ndarray, sos_list:list[np.ndarray], zi_list:list[np.ndarray]):
        """
        Apply multiple notch filters to multi-channel data in real-time using sosfilt.

        Args:
            data (ndarray): Data to be filtered (num_samples, channel).
            sos_list (list of ndarray): List of SOS arrays for each notch filter.
            zi_list (list of ndarray): List of initial filter states for each notch filter.

        Returns:
            filtered_data (ndarray): Notch filtered data.
            updated_zi_list (list of ndarray): Updated filter states for each notch filter.
        """
        # Iterate through each notch filter
        for sos, zi in zip(sos_list, zi_list):
            # Apply the notch filter to each channel
            num_channels = data.shape[1]
            for ch in range(num_channels):
                data[:, ch], zi[ch] = sosfilt(sos, data[:, ch], zi=zi[ch])
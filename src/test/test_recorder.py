import pandas as pd
import numpy as np

def compute_ema(values, period):
    """
    Compute EMA for a given array of raw values (absolute used),
    using the EMA period specified. The alpha is computed as 2/(period+1).
    """
    alpha = 2 / (period + 1)
    ema = np.zeros_like(values, dtype=float)
    ema[0] = abs(values[0])
    for i in range(1, len(values)):
        ema[i] = alpha * abs(values[i]) + (1 - alpha) * ema[i-1]
    return ema

def main(input_csv='data/s_2/ema_data_1734323631.csv', output_csv='ema_differences.csv'):
    # Read the CSV file
    df = pd.read_csv(input_csv)

    # Identify all channels from the given columns:
    # Channels appear as ch_1_raw, ch_2_raw,... Check for all columns starting with 'ch_' and ending with '_raw'
    channels = [col for col in df.columns if col.startswith('ch_') and col.endswith('_raw')]

    # For each channel, we have EMA columns like ch_X_ema_1, ch_X_ema_2, ch_X_ema_4, etc.
    # Let's find all ema columns for each channel
    ema_periods = [1,2,4,8,16,32,64]

    # Prepare a results list to store summary for each channel and period
    results = []

    for ch_raw_col in channels:
        ch_number = ch_raw_col.split('_')[1]  # e.g. ch_1_raw -> '1'
        raw_values = df[ch_raw_col].to_numpy()

        for period in ema_periods:
            ema_col_name = f"ch_{ch_number}_ema_{period}"
            if ema_col_name in df.columns:
                provided_ema = df[ema_col_name].to_numpy()
                
                # Compute our own EMA
                computed = compute_ema(raw_values, period)
                
                # Compute differences
                differences = computed - provided_ema
                avg_diff = np.mean(differences)
                max_diff = np.max(np.abs(differences))
                std_diff = np.std(differences)
                
                results.append({
                    'channel': ch_number,
                    'period': period,
                    'average_difference': avg_diff,
                    'largest_difference': max_diff,
                    'std_deviation': std_diff
                })

    # Convert results to a DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    main()

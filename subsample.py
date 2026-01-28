import pandas as pd

def subsample_csv(input_file, frequencies):
    print(f"Processing {input_file}...")
    
    # 1. Load the dataset
    df = pd.read_csv(input_file)
    
    # 2. Clean Data: Remove duplicate timestamps if they exist
    # (This is necessary because 'resample' requires a unique time index)
    df = df.drop_duplicates(subset=['ros_time'], keep='first')
    
    # 3. Create a Datetime Index
    # We convert the unix timestamp (seconds) to a datetime object
    df['datetime'] = pd.to_datetime(df['ros_time'], unit='s')
    df = df.set_index('datetime').sort_index()

    # 4. Loop through each desired frequency
    for freq in frequencies:
        # Calculate the time interval in milliseconds (e.g., 5Hz -> 200ms)
        interval_ms = int((1.0 / freq) * 1000)
        offset_str = f'{interval_ms}ms'
        
        # 5. Resample
        # .nearest(limit=1) finds the closest original row to our target time
        # This preserves exact sensor values rather than averaging them.
        df_resampled = df.resample(offset_str).nearest(limit=1)
        
        # Remove any empty rows created by gaps in data
        df_resampled = df_resampled.dropna(subset=['ros_time'])
        
        # Reset index to make 'ros_time' a normal column again
        df_resampled = df_resampled.reset_index(drop=True)
        
        # 6. Save to CSV
        output_filename = f"S1_{freq}Hz.csv"
        df_resampled.to_csv(output_filename, index=False)
        print(f" -> Created {output_filename} ({len(df_resampled)} rows)")

# Configuration
target_frequencies = [0.5] # Frequencies in Hz
# target_frequencies = [5, 3, 1, 0.1] # Frequencies in Hz
subsample_csv('csv-learning/S1_10Hz.csv', target_frequencies)
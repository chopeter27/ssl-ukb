import re
import glob
import os
import numpy as np
import pandas as pd
import pathlib
from tqdm import tqdm
import gc

# Get the absolute path to the repo root (2 levels up from this file)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Default data and output directories relative to the repo root
DATA_DIR = os.path.join(REPO_ROOT, 'data', 'capture24')  # where the .csv.gz and annotation CSV live
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'UKBB', 'capture24_30hz_w30_o0')  # processed output

# Memory-friendly settings
BATCH_SIZE = 5  # Process files in small batches
SAVE_EVERY_N_BATCHES = 10  # Save intermediate results every N batches

# don't edit below this line (unless deliberately changing parameters)
DEVICE_HZ = 100  # Hz
RESAMPLE_HZ = 30  # Hz
WINDOW_SEC = 30  # seconds
WINDOW_OVERLAP_SEC = 0  # seconds
WINDOW_LEN = int(DEVICE_HZ * WINDOW_SEC)  # device ticks
WINDOW_OVERLAP_LEN = int(DEVICE_HZ * WINDOW_OVERLAP_SEC)  # device ticks
WINDOW_STEP_LEN = WINDOW_LEN - WINDOW_OVERLAP_LEN  # device ticks
WINDOW_TOL = 0.01  # 1%
DATAFILES = os.path.join(DATA_DIR, 'P*.csv.gz')
ANNOLABELFILE = os.path.join(DATA_DIR, 'annotation-label-dictionary.csv')
LABEL = 'label:Walmsley2020'

annolabel = pd.read_csv(ANNOLABELFILE, index_col='annotation')


def resize(x, length, axis=1):
    """Resize the temporal length using linear interpolation.
    X must be of shape (N,M,C) (channels last) or (N,C,M) (channels first),
    where N is the batch size, M is the temporal length, and C is the number
    of channels.
    If X is channels-last, use axis=1 (default).
    If X is channels-first, use axis=2.
    """
    from scipy.interpolate import interp1d

    length_orig = x.shape[axis]
    t_orig = np.linspace(0, 1, length_orig, endpoint=True)
    t_new = np.linspace(0, 1, length, endpoint=True)
    x = interp1d(t_orig, x, kind="linear", axis=axis, assume_sorted=True)(
        t_new
    )
    return x


def is_good_quality(w):
    """ Window quality check """

    if w.isna().any().any():
        return False

    if len(w) != WINDOW_LEN:
        return False

    if len(w['annotation'].unique()) > 1:
        return False

    w_start, w_end = w.index[0], w.index[-1]
    w_duration = w_end - w_start
    target_duration = pd.Timedelta(WINDOW_SEC, 's')
    if np.abs(w_duration - target_duration) > WINDOW_TOL * target_duration:
        return False

    return True


def process_single_file(datafile):
    """Process a single file and yield windows one at a time"""
    try:
        # Read data with chunking if file is large
        data = pd.read_csv(datafile, parse_dates=['time'], index_col='time',
                          dtype={'x': 'f4', 'y': 'f4', 'z': 'f4', 'annotation': 'str'})
        
        p = re.search(r'(P\d{3})', datafile, flags=re.IGNORECASE).group()
        
        for i in range(0, len(data), WINDOW_STEP_LEN):
            w = data.iloc[i:i + WINDOW_LEN]
            
            if not is_good_quality(w):
                continue
            
            t = w.index[0].to_datetime64()
            x = w[['x', 'y', 'z']].values
            y = annolabel.loc[w['annotation'].iloc[0], LABEL]
            
            if DEVICE_HZ != RESAMPLE_HZ:
                x = resize(x.reshape(1, -1, 3), int(RESAMPLE_HZ * WINDOW_SEC))[0]
            
            yield x, y, t, p
            
    except Exception as e:
        print(f"Error processing {datafile}: {e}")
        return
    finally:
        # Clean up
        del data
        gc.collect()


def save_batch_data(X_batch, Y_batch, T_batch, P_batch, batch_num):
    """Save a batch of data to temporary files"""
    batch_dir = os.path.join(OUT_DIR, 'temp_batches')
    pathlib.Path(batch_dir).mkdir(parents=True, exist_ok=True)
    
    np.save(os.path.join(batch_dir, f'X_batch_{batch_num:04d}'), X_batch)
    np.save(os.path.join(batch_dir, f'Y_batch_{batch_num:04d}'), Y_batch)
    np.save(os.path.join(batch_dir, f'T_batch_{batch_num:04d}'), T_batch)
    np.save(os.path.join(batch_dir, f'P_batch_{batch_num:04d}'), P_batch)


def combine_batch_files():
    """Combine all batch files into final arrays"""
    batch_dir = os.path.join(OUT_DIR, 'temp_batches')
    
    # Get all batch files
    x_files = sorted(glob.glob(os.path.join(batch_dir, 'X_batch_*.npy')))
    y_files = sorted(glob.glob(os.path.join(batch_dir, 'Y_batch_*.npy')))
    t_files = sorted(glob.glob(os.path.join(batch_dir, 'T_batch_*.npy')))
    p_files = sorted(glob.glob(os.path.join(batch_dir, 'P_batch_*.npy')))
    
    if not x_files:
        print("No batch files found!")
        return
    
    print(f"Combining {len(x_files)} batch files...")
    
    # Process in chunks to avoid memory issues
    X_combined = []
    Y_combined = []
    T_combined = []
    P_combined = []
    
    for i, (x_file, y_file, t_file, p_file) in enumerate(zip(x_files, y_files, t_files, p_files)):
        print(f"Loading batch {i+1}/{len(x_files)}")
        
        X_batch = np.load(x_file)
        Y_batch = np.load(y_file)
        T_batch = np.load(t_file)
        P_batch = np.load(p_file)
        
        X_combined.append(X_batch)
        Y_combined.append(Y_batch)
        T_combined.append(T_batch)
        P_combined.append(P_batch)
        
        # Clean up loaded arrays
        del X_batch, Y_batch, T_batch, P_batch
        gc.collect()
    
    # Final combination
    print("Final combination...")
    X = np.vstack(X_combined)
    Y = np.hstack(Y_combined)
    T = np.hstack(T_combined)
    P = np.hstack(P_combined)
    
    # Save final results
    np.save(os.path.join(OUT_DIR, 'X'), X)
    np.save(os.path.join(OUT_DIR, 'Y_Walmsley'), Y)
    np.save(os.path.join(OUT_DIR, 'time'), T)
    np.save(os.path.join(OUT_DIR, 'pid'), P)
    
    # Clean up temporary files
    import shutil
    shutil.rmtree(batch_dir)
    
    return X, Y, T, P


if __name__ == '__main__':
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Get all data files
    datafiles = glob.glob(DATAFILES)
    print(f"Found {len(datafiles)} files to process")
    
    # Process files in batches
    X_batch, Y_batch, T_batch, P_batch = [], [], [], []
    batch_num = 0
    
    for file_idx, datafile in enumerate(tqdm(datafiles, desc="Processing files")):
        print(f"\nProcessing {os.path.basename(datafile)} ({file_idx+1}/{len(datafiles)})")
        
        # Process windows from this file
        file_windows = 0
        for x, y, t, p in process_single_file(datafile):
            X_batch.append(x)
            Y_batch.append(y)
            T_batch.append(t)
            P_batch.append(p)
            file_windows += 1
            
            # Save batch when it gets large enough
            if len(X_batch) >= 1000:  # Adjust this number based on your memory
                print(f"  Saving batch {batch_num} with {len(X_batch)} windows")
                save_batch_data(
                    np.array(X_batch), np.array(Y_batch), 
                    np.array(T_batch), np.array(P_batch), 
                    batch_num
                )
                X_batch, Y_batch, T_batch, P_batch = [], [], [], []
                batch_num += 1
                gc.collect()
        
        print(f"  Extracted {file_windows} windows from {os.path.basename(datafile)}")
    
    # Save remaining data
    if X_batch:
        print(f"Saving final batch {batch_num} with {len(X_batch)} windows")
        save_batch_data(
            np.array(X_batch), np.array(Y_batch), 
            np.array(T_batch), np.array(P_batch), 
            batch_num
        )
    
    # Combine all batch files
    print("\nCombining all batches...")
    X, Y, T, P = combine_batch_files()
    
    print(f"\nSaved in {OUT_DIR}")
    print("X shape:", X.shape)
    print("Y distribution:")
    print(pd.Series(Y).value_counts())
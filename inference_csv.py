"""
Batch process all Strath-Step CSV files with SSL+HMM inference
Reads all CSV files from the raw directory and generates activity classifications

Usage:
    python batch_inference_strath.py

Input Directory: 
    /home/peter.cho@vivosense.com/ssl-ukb/data/strath-step/Participant Data/ActiGraph Data/LW Actigraph RAW_CSV/

Output Directory:
    /home/peter.cho@vivosense.com/ssl-ukb/data/strath-step/OxWearables/actiNET/
"""

import os
import torch
import pandas as pd
import numpy as np
import gzip
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from utils.data import NormalDataset
from datetime import datetime
import glob

from models.hmm import HMM
import models.sslnet as ssl
import utils.utils as utils

# Configuration
DEVICE_HZ = 30  # Hz
WINDOW_SEC = 30  # seconds
WINDOW_OVERLAP_SEC = 0  # seconds
WINDOW_LEN = int(DEVICE_HZ * WINDOW_SEC)  # device ticks
WINDOW_OVERLAP_LEN = int(DEVICE_HZ * WINDOW_OVERLAP_SEC)  # device ticks
WINDOW_STEP_LEN = WINDOW_LEN - WINDOW_OVERLAP_LEN  # device ticks

# Paths
INPUT_DIR = Path("/home/peter.cho@vivosense.com/ssl-ukb/data/strath-step/Participant Data/ActiGraph Data/LW Actigraph RAW_CSV")
OUTPUT_DIR = Path("/home/peter.cho@vivosense.com/ssl-ukb/data/strath-step/OxWearables/actiNET")

log = utils.get_logger()

def read_actigraph_csv(input_file):
    """Read ActiGraph CSV accelerometer data with proper header handling"""
    
    # Handle gzipped files
    input_file = str(input_file)
    
    if input_file.endswith('.gz'):
        with gzip.open(input_file, 'rt') as f:
            # Skip the header lines (first 10 lines) and read the data
            data = pd.read_csv(f, skiprows=10)
    else:
        # Skip the header lines (first 10 lines) and read the data
        data = pd.read_csv(input_file, skiprows=10)
    
    # Rename columns to standard format
    data = data.rename(columns={
        'Timestamp': 'timestamp',
        'Accelerometer X': 'x',
        'Accelerometer Y': 'y', 
        'Accelerometer Z': 'z'
    })
    
    # Parse timestamp and set as index
    data['timestamp'] = pd.to_datetime(data['timestamp'], format='%m/%d/%Y %H:%M:%S.%f')
    data.set_index('timestamp', inplace=True)
    
    # Keep only x, y, z columns
    data = data[['x', 'y', 'z']].copy()
    
    # Remove any NaN values
    data = data.dropna()
    
    # The original data is at 100Hz, but we need 30Hz for the model
    # Resample to 30Hz (every ~33.33ms)
    data = data.resample('33.33ms').mean()
    data = data.dropna()
    
    log.info(f"Processed data: {len(data)} points at 30Hz")
    log.info(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Create basic info dict
    info = {
        'Filename': str(input_file),
        'Device': 'ActiGraph GT3X+',
        'ReadErrors': 0,
        'SampleRate': 30,  # After resampling
        'NumTicks': len(data),
        'CalibOK': 1,
        'ReadOK': 1
    }
    
    return data, info

def vectorized_stride_v2(acc, time, window_size, stride_size):
    """Numpy vectorised windowing with stride"""
    start = 0
    max_time = len(time)

    sub_windows = (start +
                   np.expand_dims(np.arange(window_size), 0) +
                   np.expand_dims(np.arange(max_time + 1, step=stride_size), 0).T
                   )[:-1]

    return acc[sub_windows], time[sub_windows]

def df_to_windows(df):
    """Convert a time series dataframe to a windowed Numpy array"""
    acc = df[['x', 'y', 'z']].to_numpy()
    time = df.index.to_numpy()

    # convert to windows
    x, t = vectorized_stride_v2(acc, time, WINDOW_LEN, WINDOW_STEP_LEN)

    # drop the whole window if it contains a NaN
    na = np.isnan(x).any(axis=1).any(axis=1)
    x = x[~na]
    t = t[~na]

    return x, t[:, 0]

def extract_participant_id(filename):
    """Extract participant ID from Strath-Step filename format"""
    # From FLAC_AG_LW_7111_7DRAW.csv extract 7111
    stem = Path(filename).stem
    if 'FLAC_AG_LW_' in stem:
        return stem.split('_')[3]  # Gets 7111 from FLAC_AG_LW_7111_7DRAW
    else:
        return stem.split('_')[0]  # fallback

def find_csv_files():
    """Find all CSV files in the input directory"""
    csv_files = list(INPUT_DIR.glob("*.csv"))
    if not csv_files:
        log.error(f"No CSV files found in {INPUT_DIR}")
        return []
    
    log.info(f"Found {len(csv_files)} CSV files to process")
    return csv_files

def process_single_file(input_file, sslnet, hmm_ssl, my_device):
    """Process a single CSV file and return results"""
    
    input_path = Path(input_file)
    pt_id = extract_participant_id(input_file)
    
    log.info(f"Processing {input_path.name} -> Participant {pt_id}")
    
    try:
        # Load and process data
        data, info = read_actigraph_csv(input_file)
        
        # Store original start/end times for reindexing later
        data_start = data.index[0]
        data_end = data.index[-1]
        
        # Prepare dataset
        log.info('Windowing')
        X, T = df_to_windows(data)
        del data  # free up memory
        
        if len(X) == 0:
            log.warning(f"No valid windows found for {pt_id}")
            return None
        
        dataset = NormalDataset(X, name=pt_id)
        dataloader = DataLoader(dataset, batch_size=120, shuffle=False, num_workers=0)
        
        # Do inference
        log.info('SSL inference')
        _, y_prob, _ = ssl.predict(sslnet, dataloader, my_device, output_logits=True)
        y_pred = np.argmax(y_prob, axis=1)
        
        log.info('HMM smoothing')
        y_pred_hmm = hmm_ssl.viterbi(y_pred)
        
        # Create results dataframe
        df_results = pd.DataFrame({
            'timestamp': T,
            'activity': utils.le.inverse_transform(y_pred_hmm)
        })
        df_results['timestamp'] = pd.to_datetime(df_results['timestamp'])
        df_results.set_index('timestamp', inplace=True)
        
        # Reindex for missing values
        newindex = pd.date_range(data_start, data_end, freq=f'{WINDOW_SEC}S')
        df_results = df_results.reindex(newindex, method='nearest', fill_value='missing')
        
        # Reset index to have timestamp as column for CSV output
        df_results.reset_index(inplace=True)
        
        return df_results, pt_id
        
    except Exception as e:
        log.error(f"Error processing {input_file}: {e}")
        return None

def main():
    start_time = datetime.now()
    
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Load config
    cfg = OmegaConf.load("conf/config.yaml")
    
    # Setup device
    GPU = cfg.gpu
    if GPU != -1:
        my_device = "cuda:" + str(GPU)
    else:
        my_device = "cpu"
    
    log.info(f"Using device: {my_device}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory: {OUTPUT_DIR}")
    
    # Find all CSV files
    csv_files = find_csv_files()
    if not csv_files:
        return
    
    # Load pretrained models once
    log.info("Loading pretrained SSL model...")
    sslnet = ssl.get_sslnet(my_device, cfg, load_weights=True)
    
    log.info("Loading pretrained HMM...")
    hmm_ssl = HMM(utils.classes, uniform_prior=cfg.hmm.uniform_prior)
    hmm_ssl.load(cfg.hmm.weights_ssl)
    
    # Process each file
    processed_count = 0
    failed_count = 0
    
    for i, csv_file in enumerate(csv_files, 1):
        log.info(f"\n[{i}/{len(csv_files)}] Processing {csv_file.name}")
        
        # Check if output already exists
        pt_id = extract_participant_id(csv_file)
        output_file = OUTPUT_DIR / f"{pt_id}.csv"
        
        if output_file.exists():
            log.info(f"Output already exists for {pt_id}, skipping...")
            continue
        
        # Process file
        result = process_single_file(csv_file, sslnet, hmm_ssl, my_device)
        
        if result is not None:
            df_results, pt_id = result
            
            # Save results
            df_results.to_csv(output_file, index=False)
            log.info(f"Saved results to {output_file}")
            processed_count += 1
            
        else:
            log.error(f"Failed to process {csv_file.name}")
            failed_count += 1
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    log.info(f"\n=== Batch Processing Complete ===")
    log.info(f"Total files found: {len(csv_files)}")
    log.info(f"Successfully processed: {processed_count}")
    log.info(f"Failed: {failed_count}")
    log.info(f"Total duration: {duration}")
    log.info(f"Output directory: {OUTPUT_DIR}")
    
    # List generated files
    output_files = list(OUTPUT_DIR.glob("*.csv"))
    log.info(f"\nGenerated {len(output_files)} output files:")
    for f in sorted(output_files):
        log.info(f"  {f.name}")

if __name__ == "__main__":
    main()
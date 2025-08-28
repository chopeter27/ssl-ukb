"""
Batch process all Strath-Step CSV files with SSL+HMM inference
"""

import os
from pathlib import Path
import subprocess
import sys

def find_csv_files(data_dir):
    """Find all CSV files in the Strath-Step data directory"""
    csv_files = []
    search_path = Path(data_dir) / "Participant Data" / "ActiGraph Data" / "LW Actigraph RAW_CSV"
    
    if search_path.exists():
        csv_files = list(search_path.glob("*.csv"))
    
    return csv_files

def main():
    data_dir = Path("/home/peter.cho@vivosense.com/ssl-ukb/data/strath-step")
    
    # Find all CSV files
    csv_files = find_csv_files(data_dir)
    
    if not csv_files:
        print("No CSV files found!")
        sys.exit(1)
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Process each file
    for i, csv_file in enumerate(csv_files, 1):
        print(f"\n[{i}/{len(csv_files)}] Processing {csv_file.name}")
        
        try:
            subprocess.run([
                "python", "inference_csv.py", 
                str(csv_file)
            ], check=True)
            print(f"✓ Successfully processed {csv_file.name}")
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to process {csv_file.name}: {e}")
            continue
    
    print(f"\n🎉 Batch processing complete!")
    print(f"Results saved to: {data_dir}/OxWearables/actiNET/")

if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""download_strath_step_data.py

Utility script for downloading Strath-Step ActiGraph data.

- Reads a base SAS URL (without container name) from .env as CSU_AFFECT_SAS_URL.
- Appends the strath-step container name dynamically.
- Downloads specific CSV files listed in ActiNET_ACC.txt into data/strath-step/, 
  preserving directory structure.

"""
from pathlib import Path
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from azure.storage.blob import ContainerClient


# --------------------------------------------------------------------------- #
#                         CONFIGURATION                                       #
# --------------------------------------------------------------------------- #
container_name = "strath-step"

# Load environment and build full SAS URL with container name
load_dotenv()
base_sas_url = os.getenv("CSU_AFFECT_SAS_URL")  # base should exclude container name
if not base_sas_url:
    sys.exit("Environment variable 'CSU_AFFECT_SAS_URL' not set. Add it to a .env file.")

# Ensure no duplicate '?' when appending container
if '?' in base_sas_url:
    base, query = base_sas_url.split('?', 1)
    sas_url = f"{base}/{container_name}?{query}"
else:
    sas_url = f"{base_sas_url}/{container_name}"

DATA_DIR = Path("/home/peter.cho@vivosense.com/ssl-ukb/data/strath-step")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Path to the file list
FILE_LIST_PATH = Path("ActiNET_ACC.txt")
# --------------------------------------------------------------------------- #


# --------------------------- AZURE CONNECTION ------------------------------ #
try:
    container_client = ContainerClient.from_container_url(sas_url)
except ValueError as exc:
    sys.exit(f"[error] Failed to create ContainerClient: {exc}")


def download_blob_to_file(blob_name: str, dest_path: Path) -> None:
    '''Download *blob_name* to *dest_path* unless it already exists.'''
    if dest_path.exists():
        print(f"[skip] {dest_path.name} already exists.")
        return

    # Create parent directories if they don't exist
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    blob_client = container_client.get_blob_client(blob_name)
    print(f"[dl ] {blob_name} -> {dest_path}")
    try:
        with open(dest_path, "wb") as fh:
            fh.write(blob_client.download_blob().readall())
        print(f"[✓ ] Successfully downloaded {dest_path.name}")
    except Exception as e:
        print(f"[error] Failed to download {blob_name}: {e}")


def read_file_list() -> list[str]:
    '''Read the list of files to download from ActiNET_ACC.txt'''
    if not FILE_LIST_PATH.exists():
        sys.exit(f"[error] File list not found: {FILE_LIST_PATH}")
    
    with open(FILE_LIST_PATH, 'r') as f:
        files = [line.strip() for line in f if line.strip()]
    
    print(f"[info] Found {len(files)} files to download from {FILE_LIST_PATH}")
    return files


def main() -> None:
    # ---------------------- Read file list ---------------------------------- #
    files_to_download = read_file_list()
    
    # ---------------------------- Download ---------------------------------- #
    for blob_path in files_to_download:
        # Create local path preserving directory structure
        # Remove the container prefix if it exists in the file list
        if blob_path.startswith('strath-step/'):
            blob_path = blob_path[len('strath-step/'):]
        
        # Create destination path
        dest_path = DATA_DIR / blob_path
        
        # Download the file
        download_blob_to_file(blob_path, dest_path)
    
    print(f"[✔] Download complete. Files saved to {DATA_DIR}")


if __name__ == "__main__":
    main()
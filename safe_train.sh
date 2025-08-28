#!/bin/bash

# Configuration
PROJECT_DIR="/home/peter.cho@vivosense.com/ssl-ukb"
PYTHON_PATH="/home/peter.cho@vivosense.com/ssl-ukb/venv/bin/python"
LOG_FILE="training_$(date +%Y%m%d_%H%M%S).log"
MAX_MEMORY_PERCENT=85  # Stop if system memory usage exceeds this
MIN_FREE_MEMORY_GB=2   # Minimum free memory required to start

# Function to check memory
check_memory() {
    local mem_percent=$(free | grep '^Mem:' | awk '{printf("%.0f", $3/$2 * 100.0)}')
    local free_gb=$(free -m | grep '^Mem:' | awk '{printf("%.1f", $7/1024)}')
    
    echo "Memory usage: ${mem_percent}%, Free: ${free_gb}GB" | tee -a "$LOG_FILE"
    
    if [ "$mem_percent" -gt "$MAX_MEMORY_PERCENT" ]; then
        echo "WARNING: Memory usage too high (${mem_percent}% > ${MAX_MEMORY_PERCENT}%)" | tee -a "$LOG_FILE"
        return 1
    fi
    
    if (( $(echo "$free_gb < $MIN_FREE_MEMORY_GB" | bc -l) )); then
        echo "WARNING: Insufficient free memory (${free_gb}GB < ${MIN_FREE_MEMORY_GB}GB)" | tee -a "$LOG_FILE"
        return 1
    fi
    
    return 0
}

# Function to log system info
log_system_info() {
    echo "=== SYSTEM INFO AT $(date) ===" | tee -a "$LOG_FILE"
    echo "Working directory: $(pwd)" | tee -a "$LOG_FILE"
    echo "Python path: $PYTHON_PATH" | tee -a "$LOG_FILE"
    free -h | tee -a "$LOG_FILE"
    echo "Disk usage:" | tee -a "$LOG_FILE"
    df -h "$PROJECT_DIR" | tee -a "$LOG_FILE"
    echo "GPU usage:" | tee -a "$LOG_FILE"
    nvidia-smi 2>/dev/null | tee -a "$LOG_FILE" || echo "No GPU info available" | tee -a "$LOG_FILE"
    echo "===============================" | tee -a "$LOG_FILE"
}

# Main execution
cd "$PROJECT_DIR" || { echo "Failed to change to project directory"; exit 1; }

echo "Starting safe training script" | tee "$LOG_FILE"

# Initial system check
log_system_info

# Check if we have enough memory to start
if ! check_memory; then
    echo "Not enough memory to start safely. Exiting." | tee -a "$LOG_FILE"
    exit 1
fi

# Set memory limits to prevent complete system freeze
ulimit -v $(($(free | grep '^Mem:' | awk '{print $2}') * 80 / 100))  # Limit to 80% of total memory

echo "Starting training at $(date)" | tee -a "$LOG_FILE"

# Run the training script (no timeout - let it run as long as needed)
"$PYTHON_PATH" train.py 2>&1 | tee -a "$LOG_FILE"
EXIT_CODE=$?

echo "Training completed at $(date) with exit code: $EXIT_CODE" | tee -a "$LOG_FILE"

# Final system status
log_system_info

# Check for common error patterns
if [ $EXIT_CODE -eq 137 ] || [ $EXIT_CODE -eq 9 ]; then
    echo "Process was killed (likely out of memory)" | tee -a "$LOG_FILE"
elif [ $EXIT_CODE -eq 130 ]; then
    echo "Process was interrupted (SIGINT/Ctrl+C)" | tee -a "$LOG_FILE"
elif [ $EXIT_CODE -ne 0 ]; then
    echo "Process failed with error code $EXIT_CODE" | tee -a "$LOG_FILE"
else
    echo "Training completed successfully!" | tee -a "$LOG_FILE"
fi

echo "Log saved to: $LOG_FILE"
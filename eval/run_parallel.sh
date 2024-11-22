#!/bin/bash

# Configuration
TOTAL_PROBLEMS=3000  # Total number of problems to process
NUM_CORES=$(nproc)
NUM_PROCESSES=$((NUM_CORES - 1))  # Leave one core free for system processes
CHUNK_SIZE=$((TOTAL_PROBLEMS / NUM_PROCESSES))

# Add command line argument parsing for directory
if [ $# -eq 0 ]; then
    echo "Error: Directory argument is required"
    echo "Usage: $0 <directory>"
    exit 1
fi
DIRECTORY="$1"

echo "Detected $NUM_CORES cores, using $NUM_PROCESSES processes"

# Create results directory if it doesn't exist
mkdir -p results

# Launch processes in parallel
for i in $(seq 0 $((NUM_PROCESSES-1))); do
    START=$((i * CHUNK_SIZE))
    END=$((START + CHUNK_SIZE))
    
    # For the last chunk, process remaining problems
    if [ $i -eq $((NUM_PROCESSES-1)) ]; then
        END=$TOTAL_PROBLEMS
    fi
    
    echo "Launching process for range $START-$END"
    python test_one_solution_single.py \
        --start $START \
        --end $END \
        --save $DIRECTORY &
done

# Wait for all background processes to complete
wait

# Combine results (optional)
python combine_results.py $DIRECTORY
# You might want to write a small Python script to combine the individual JSON files
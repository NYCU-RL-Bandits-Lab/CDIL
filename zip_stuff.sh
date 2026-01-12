#!/bin/bash

# Name of the output zip file (saved in your current directory, not inside tfboard)
OUTPUT_ZIP="../latest_logs.zip"

# 1. Change directory to tfboard so the zip structure starts at 'avatardice'
cd tfboard || { echo "Directory tfboard not found"; exit 1; }

# 2. Loop through Seeds (3 and 4)
for SEED in 3 4; do
    # 3. Loop through the environment names
    for ENV in hopper wipe ant cheetah door lift; do
        
        # Construct the directory path
        DIR="avatardice/${ENV}_seed${SEED}"
        
        # Check if directory exists
        if [ -d "$DIR" ]; then
            # Get the latest file in the directory (ls -t sorts by time, head -n 1 takes the first)
            LATEST_FILE=$(ls -t "$DIR" | head -n 1)
            
            if [ -n "$LATEST_FILE" ]; then
                echo "Adding: $DIR/$LATEST_FILE"
                # Add to zip (zip automatically appends if the archive exists)
                zip "$OUTPUT_ZIP" "$DIR/$LATEST_FILE"
            else
                echo "Warning: No files found in $DIR"
            fi
        else
            echo "Warning: Directory $DIR does not exist"
        fi
    done
done

echo "Done! Archive saved to $OUTPUT_ZIP"
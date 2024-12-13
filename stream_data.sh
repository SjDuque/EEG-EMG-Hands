#!/bin/bash
# RUNS ON MAC OS ONLY
# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Path to the virtual environment's Python
PYTHON="$SCRIPT_DIR/.conda/bin/python"

# Path to the Python scripts (relative to the script location)
SCRIPT_1="$SCRIPT_DIR/src/hand_tracking/mediapipe_server_lsl.py"
SCRIPT_2="$SCRIPT_DIR/src/hand_tracking/mediapipe_client_lsl.py"
SCRIPT_3="$SCRIPT_DIR/src/openbci/graph_brainflow.py" 

# Open the first script in a new terminal
osascript -e "tell application \"Terminal\" to do script \"$PYTHON $SCRIPT_1\""

# Open the second script in another new terminal
osascript -e "tell application \"Terminal\" to do script \"$PYTHON $SCRIPT_2\""

# Open the third script in another new terminal
osascript -e "tell application \"Terminal\" to do script \"$PYTHON $SCRIPT_3\""

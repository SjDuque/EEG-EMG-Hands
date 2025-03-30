#!/bin/bash
# RUNS ON MAC OS ONLY
# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Path to the virtual environment's Python
PYTHON=".conda/bin/python"

# Path to the Python scripts (relative to the script location)
SCRIPT_1="src/hand_tracking/mediapipe_server.py"
SCRIPT_2="src/hand_tracking/mediapipe_client.py"
SCRIPT_3="src/exg/brainflow_iir_server.py" 
SCRIPT_4="src/exg/brainflow_client.py"

# Open the first script in a new terminal
osascript -e "tell application \"Terminal\" to do script \"cd $SCRIPT_DIR && $PYTHON $SCRIPT_1\""

# Open the second script in another new terminal
osascript -e "tell application \"Terminal\" to do script \"cd $SCRIPT_DIR && $PYTHON $SCRIPT_2\""

# Open the third script in another new terminal
osascript -e "tell application \"Terminal\" to do script \"cd $SCRIPT_DIR && $PYTHON $SCRIPT_3\""

# Open the fourth script in another new terminal
osascript -e "tell application \"Terminal\" to do script \"cd $SCRIPT_DIR && $PYTHON $SCRIPT_4\""

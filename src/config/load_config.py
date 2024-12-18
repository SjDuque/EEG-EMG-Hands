import json

# Load the JSON data from the file
def load_finger_thresholds():
    with open('config/finger_thresholds.json', 'r') as file:
        data = json.load(file)

    # Convert lists back to tuples if necessary
    finger_thresholds = tuple(tuple(pair) for pair in data['finger_thresholds'])

    return finger_thresholds
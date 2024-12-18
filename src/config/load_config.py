import json

# Load the JSON data from the file
def load_finger_thresholds():
    with open('config/finger_thresholds.json', 'r') as file:
        data = json.load(file)

    # Convert lists back to tuples if 
    finger_names = tuple(data['finger_names'])
    finger_thresholds = tuple(tuple(pair) for pair in data['finger_thresholds'])

    return finger_names, finger_thresholds
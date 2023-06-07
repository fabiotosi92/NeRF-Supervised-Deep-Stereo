import numpy as np
import json
import argparse
import shutil

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Argument parser')
parser.add_argument('--src_json_file', dest='src_json_file', type=argparse.FileType('r'), default='transform.json',
                    help='path to the transform json file')
parser.add_argument('--dest_json_file', dest='dest_json_file', type=argparse.FileType('w'), default='transform_left.json',
                    help='path to the transform json file')
parser.add_argument('--shift', type=float, help='shift', default=-0.2)
args = parser.parse_args()

def get_translated_matrix(transform_matrix: list, shift: float):
    # Convert transform_matrix to numpy array
    transform_matrix = np.array(transform_matrix)

    # Create shift matrix
    shift_matrix = np.array([[1.0, 0.0, 0.0, shift],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0]])

    # Calculate the translated matrix
    return np.linalg.inv(shift_matrix @ np.linalg.inv(transform_matrix)).tolist()

# Open the destination file
with args.dest_json_file as dest_file:
    # Open the source JSON file
    with args.src_json_file as src_file:
        # Load the JSON data
        data = json.load(src_file)
    
    # Iterate through the frames in the JSON data
    for i in data['frames']:
        t_matrix = i["transform_matrix"]
        i["transform_matrix"] = get_translated_matrix(transform_matrix=t_matrix, shift=args.shift)

    # Write the modified JSON data to the destination file
    json.dump(data, dest_file)

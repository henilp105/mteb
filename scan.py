import os
import re

def find_name_values_in_files(directory):
    name_pattern = re.compile(r'name="([^"]+)"')
    name_values = []

    # Walk through all files in the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    # Find all occurrences of the pattern in the file
                    matches = name_pattern.findall(content)
                    name_values.extend(matches)
            except Exception as e:
                print(f"Could not read file {file_path}: {e}")

    return name_values

# Specify the directory you want to search
directory_path = 'mteb/tasks/Retrieval/'

# Get the list of name values
name_values_list = find_name_values_in_files(directory_path)

# Print the list of name values
print(name_values_list)

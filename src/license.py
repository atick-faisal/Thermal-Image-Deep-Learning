import os
import datetime

LICENSE_HEADER = """# Copyright (c) {year} Atick Faisal
# Licensed under the MIT License - see LICENSE file for details
"""


def add_license_to_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    if 'Copyright' not in content:
        year = datetime.datetime.now().year
        with open(file_path, 'w') as f:
            f.write(LICENSE_HEADER.format(year=year) + '\n' + content)


def process_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                add_license_to_file(file_path)


if __name__ == '__main__':
    project_dir = '.'  # Current directory
    process_directory(project_dir)
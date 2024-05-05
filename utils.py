"""
Tools for internal code processing
"""
import yaml

def parse_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
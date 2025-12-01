import sys
import os

# Determine the parent directory (project_root)
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)

sys.path.append(parent_directory)
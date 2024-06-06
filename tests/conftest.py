# conftest.py
import sys
from pathlib import Path

# Append the directory above 'tests' to sys.path
sys.path.insert(0, str(Path(__file__).parents[1]))

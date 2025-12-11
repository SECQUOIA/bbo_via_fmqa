"""
Pytest configuration for bbo_via_fmqa tests
"""

import sys
import os

# Add parent directories to path for imports
# Add bbo_via_fmqa package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add fmqa package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'fmqa')))

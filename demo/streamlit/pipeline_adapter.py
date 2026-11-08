"""Compatibility import for the original Streamlit launcher."""
import sys
from drill_ir.demo import pipeline_adapter as _implementation
sys.modules[__name__] = _implementation

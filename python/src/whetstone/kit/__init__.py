"""The kit: supporting objects that problems hand to solutions.

    from whetstone.kit import TinyLM

Kit objects are small and meant to be read. Anything with instrumentation exposes a
``counters`` attribute for the cost judge.
"""

from whetstone.kit.counters import Counters
from whetstone.kit.simmodel import SimModel
from whetstone.kit.tinylm import KVCache, TinyLM

__all__ = ["Counters", "KVCache", "SimModel", "TinyLM"]

"""
dsproc is a toolkit for the processing of digital signals. It contains the following important classes:
    - Message - contains functions for encoding and decoding of bits, including compression, forward error correction
                and randomisers.
    - Mod - Used to modulate data into radio waves
    - Demod - Used to demodulate data from radio waves

For more information on a specific class, call help(dpsroc.Class), e.g. help(dsproc.Mod).

To get started there is an examples folder that contains example programs
"""

__version__ = '0.0.0b'
from .sig.mod import Mod
from .sig.demod import Demod
from .util.utils import AWGN, create_message
from .message.message import Message
from .sig.constellation import Constellation
from .sig.filter import Filter


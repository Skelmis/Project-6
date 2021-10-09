from dataclasses import dataclass

import numpy as np


@dataclass
class Face:
    percent: float = None
    top_left_x: int = None
    top_left_y: int = None
    bottom_right_x: int = None
    bottom_right_y: int = None

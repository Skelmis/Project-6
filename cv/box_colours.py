from enum import Enum
from typing import Tuple


class BoxColors(Enum):
    """
    Notes
    -----
    NOT RGB, BGR

    UNKNOWN:
        Doesn't know what this is about
        White
    THREAT:
        Someone deemed a threat
        Red
    BOX_KNOWLEDGE:
        Know's about the boxes existence
        Yellow
    ANALOG_INTERFACE:
        Has the ability to communicate with the box
        Black with yellow parts
    ASSET_CATALYST
        Government people? Not sure
        Blue
    """

    UNKNOWN = 0
    THREAT = 1
    BOX_KNOWLEDGE = 2
    ANALOG_INTERFACE = 3
    ASSET_CATALYST = 4

    @staticmethod
    def get_yellow() -> Tuple[int, int, int]:
        """Returns yellow"""
        return 1, 216, 255

    def get_bgr(self) -> Tuple[int, int, int]:
        """Given an enum, return the relevant rgb color"""
        if self.value == 1:
            # Red
            return 4, 0, 221

        elif self.value == 2:
            # Yellow
            return self.get_yellow()

        elif self.value == 3:
            # Black
            return 3, 2, 5

        elif self.value == 4:
            return 255, 32, 0

        # Unknown
        return 255, 255, 255

    def compare_enum(self, other: "BoxColors") -> "BoxColors":
        """
        Given two BoxColors, return the 'True' value.
        This is essentially anything but BoxColors.UNKNOWN
        """
        if self.value == 0:
            return other

        return self

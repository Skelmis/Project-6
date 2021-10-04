import itertools
import logging
import os
from pathlib import Path
from typing import Optional

import cv2

from bot_base import BotBase

log = logging.getLogger(__name__)


class CV:
    def __init__(self, bot):
        self.bot: BotBase = bot
        self.feed = cv2.VideoCapture(0)
        self.cwd = str(Path(__file__).parents[0])

        self.image_suffix = itertools.count().__next__

    def take_picture(self, image_name: str = None) -> Optional[str]:
        """Takes a picture and saves it

        Returns
        -------
        A path to the image
        """
        ret, frame = self.feed.read()
        if not ret:
            log.warning("No video feed active to capture from.")
            return None

        image_name = image_name or f"image_{self.image_suffix()}"
        image_name += ".png"

        path = os.path.join(self.cwd, "pictures", image_name)
        cv2.imwrite(path, frame)

        self.feed.release()

        return path

    def cleanup(self) -> None:
        """Cleans up video objects before shutdown"""
        self.feed.release()

        # Destroy all the windows
        cv2.destroyAllWindows()

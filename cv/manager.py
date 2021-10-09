import asyncio
import itertools
import os
from pathlib import Path
from typing import List

import cv2
import nextcord
import numpy as np

from bot_base import BotBase
from cv import CV
from cv.facial_recognition.face import Face


class Manager:
    """
    A simple class for tracking collections of photos
    """

    def __init__(self, bot):
        self.cv: CV = bot.cv
        self.bot: BotBase = bot
        self.current_group = []
        self.cwd = str(Path(__file__).parents[0])

        self.image_suffix = itertools.count().__next__

    async def handle_new(self, image: np.ndarray) -> None:
        """
        Given a new image, check to see if we have notable changes.
        If we do, either start a new group or add to the existing one.
        If not, check if we *had* a group and end it
        """
        faces: List[Face] = self.cv.face.find_face(image)
        if not faces:
            if not self.current_group:
                return

            # No more faces, time to end it
            await self.end_current_group()

        # Draw the faces
        self.cv.face.draw_faces(image, faces)

        self.current_group.append(image)

    async def end_current_group(self):
        """Called once a group is all finished and we can update discord"""
        if not bool(self.current_group):
            # Don't need to do anything
            return

        time = nextcord.utils.utcnow()
        channel = await self.bot.get_or_fetch_channel(self.bot.project_6_id)  # noqa
        initial_message: nextcord.Message = await channel.send(
            embed=nextcord.Embed(description="Movement detectedd", timestamp=time)
        )
        thread: nextcord.Thread = await initial_message.create_thread(
            name=f"Image collection"
        )

        image_suffix = self.image_suffix()
        save_path = os.path.join(self.cwd, "groups", str(image_suffix))

        # Ensure save dir exists
        Path(save_path).mkdir(parents=True)

        # Send all the images to discord
        for count, image in enumerate(self.current_group):
            image_name = f"{image_suffix}.png"
            image_path = os.path.join(save_path, image_name)

            cv2.imwrite(image_path, image)
            file = nextcord.File(image_path)
            await thread.send(file=file)
            await asyncio.sleep(0.5)

        await thread.send("That is all for this image group")

        # Clean up after we are done
        self.current_group = []

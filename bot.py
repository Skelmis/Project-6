import asyncio
import logging
import os
from pprint import pprint
from typing import List

import numpy as np
import cv2
import discord
from discord.ext import commands

from bot_base import BotBase
from cv import CV, BoxColors, RecognizeReturn

bot = BotBase(
    command_prefix=".",
    mongo_url=os.environ["MONGO_URL"],
    mongo_database_name="project_6",
)
bot.cv = CV(bot)
bot.project_6_id = 894497833053462529


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@bot.event
async def on_ready():
    chan = await bot.get_or_fetch_channel(bot.project_6_id)
    log.info("Up and ready to work")


@bot.command(aliases=["picture"])
async def pic(ctx) -> None:
    """Takes a picture and sends it to discord"""
    image = bot.cv.take_picture()
    picture_path: str = bot.cv.save_picture(image)

    file = discord.File(picture_path)
    await ctx.send("Imagine taking pictures of people. Tut.", file=file)


@bot.command(aliases=["comparepicture"])
async def com(ctx):
    image = bot.cv.take_picture()
    path_one = bot.cv.save_picture(image)
    file = discord.File(path_one)
    # await ctx.send(file=file)
    print("Taken first")

    await asyncio.sleep(10)
    image_two = bot.cv.take_picture()
    path_two = bot.cv.save_picture(image_two)
    print("Taken second")

    file = discord.File(path_two)
    # await ctx.send(file=file)

    diff_score, actual_diffs = bot.cv.compare_images(image, image_two)
    thresh, cnts = bot.cv.get_image_differences(actual_diffs)

    # loop over the contours
    for c in cnts:
        pprint(c)
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(image_two, (x, y), (x + w, y + h), (0, 0, 255), 2)

    await ctx.send("Here is the output for differences")
    # show the output images
    path = bot.cv.save_picture(image)
    file = discord.File(path)
    await ctx.send(file=file)

    path = bot.cv.save_picture(image_two)
    file = discord.File(path)
    await ctx.send(file=file)

    path = bot.cv.save_picture(actual_diffs)
    file = discord.File(path)
    await ctx.send(file=file)

    path = bot.cv.save_picture(thresh)
    file = discord.File(path)
    await ctx.send(file=file)

    await ctx.send(":thumbsup:")


@bot.command()
async def diff(ctx):
    """Given two images, return a box around the differences"""
    image_one = bot.cv.take_picture()
    await asyncio.sleep(5)
    image_two = bot.cv.take_picture()

    diff_score, actual_diffs = bot.cv.compare_images(image_one, image_two)
    if diff_score > float(0.9):
        await ctx.send(
            f"These images appear too similar to figure out, will try regardless.\nSSIM: {diff_score}"
        )

    _, contours = bot.cv.get_image_differences(actual_diffs)
    # pprint(contours)

    # Loop over all contours and get all
    # which are over 50 x 50 pixels
    relevant_contours = bot.cv.get_relevant_differences(contours)

    bot.cv.draw_bounding_boxes(image_one, relevant_contours, BoxColors.ASSET_CATALYST)
    bot.cv.draw_bounding_boxes(image_two, relevant_contours, BoxColors.ANALOG_INTERFACE)

    path = bot.cv.save_picture(image_one)
    file = discord.File(path)
    await ctx.send(file=file)

    path = bot.cv.save_picture(image_two)
    file = discord.File(path)
    await ctx.send(file=file)


@bot.command()
async def showdiff(ctx):
    """Shows only the differences"""
    image_one = bot.cv.take_picture()
    await asyncio.sleep(5)
    image_two = bot.cv.take_picture()

    diff_score, actual_diffs = bot.cv.compare_images(image_one, image_two)
    _, contours = bot.cv.get_image_differences(actual_diffs)

    # Loop over all contours and get all
    # which are over 50 x 50 pixels
    relevant_contours = bot.cv.get_relevant_differences(contours)

    await ctx.send("Here are all the changes from within image_one")
    subset_images: List[np.ndarray] = bot.cv.get_subset_images(
        image_one, relevant_contours
    )
    for image in subset_images:
        path = bot.cv.save_picture(image)
        file = discord.File(path)
        await ctx.send(file=file)

    await ctx.send("Here are the same regions in image two")
    subset_images: List[np.ndarray] = bot.cv.get_subset_images(
        image_two, relevant_contours
    )
    for image in subset_images:
        path = bot.cv.save_picture(image)
        file = discord.File(path)
        await ctx.send(file=file)


@bot.command()
async def faceboxes(ctx):
    """Attempts to detect a face"""
    image = bot.cv.take_picture()
    path = bot.cv.save_picture(image)
    file = discord.File(path)
    await ctx.send("Reference image", file=file)

    # DNN stuff
    bot.cv.face.find_face(image)

    # Now image
    path = bot.cv.save_picture(image)
    file = discord.File(path)
    await ctx.send("After DNN", file=file)


@bot.command()
async def recognize(ctx):
    """Attempts to name a detected face"""
    initial_image = bot.cv.take_picture()
    path = bot.cv.save_picture(initial_image)
    file = discord.File(path)
    await ctx.send("Reference image", file=file)

    # SVC stuff
    data: RecognizeReturn = bot.cv.face.recognize(initial_image)  # noqa
    image = data.image

    bot.cv.draw_face_box(
        image,
        data.top_left_x,
        data.top_left_y,
        data.bottom_right_x,
        data.bottom_right_y,
        BoxColors.from_name(data.name),
        data.name,
    )

    # Now image
    path = bot.cv.save_picture(image)
    file = discord.File(path)
    await ctx.send("After SVC", file=file)


@bot.command(aliases=["l"])
@commands.is_owner()
async def logout(ctx) -> None:
    """Gracefully shuts the bot down"""
    bot.cv.release()
    await ctx.send("Logging out")
    await bot.close()


bot.run(os.environ["TOKEN"])

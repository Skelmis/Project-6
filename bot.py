import logging
import os
from pprint import pprint

import discord
from discord.ext import commands

from bot_base import BotBase
from cv import CV

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


@bot.command()
@commands.is_owner()
async def logout(ctx) -> None:
    """Gracefully shuts the bot down"""
    bot.cv.cleanup()
    await ctx.send("Logging out")
    await bot.close()


bot.run(os.environ["TOKEN"])

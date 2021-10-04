import logging
import os
from pprint import pprint

from bot_base import BotBase

bot = BotBase(
    command_prefix="!",
    mongo_url=os.environ["MONGO_URL"],
    mongo_database_name="project_6",
)
bot.project_6_id = 894497833053462529

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@bot.event
async def on_ready():
    log.info("Up and ready to work")


bot.run(os.environ["TOKEN"])

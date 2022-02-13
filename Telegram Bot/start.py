import ImageHandlerBot
import logging

if __name__ == "__main__":
    TOKEN = '5119349841:AAE9pb-JxkjgXOI8ZEtqDGlRhSS0-nZoHks'

    # Remove comments to Start logging
    #logging.basicConfig(level=logging.DEBUG,
    #                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    dialog_bot = ImageHandlerBot.ImageHandlerBot(TOKEN)
    dialog_bot.start()

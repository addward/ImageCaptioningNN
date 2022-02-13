from telegram.ext import Updater
from telegram.ext import MessageHandler, Filters, CallbackContext
from telegram import Document, PhotoSize, Message

import os
import concurrent.futures
import collections

import torch
import torchvision.models as models

import Caption_Net


class ImageHandlerBot(object):
    """
    Class for Telegram bot, implemented with Python Telegram Bot
    To create and start bot:
        dialog_bot = ImageHandlerBot.ImageHandlerBot(TOKEN, IMAGE_SIZE)
        dialog_bot.start()
    After that bot will handle messages using handle_message method and
    answer to user due to dialog generator function
    """

    def __init__(self, token):
        """
        Class constructor
        Parameters
        ----------
        token : unique token to control telegram bot
        image_size : size of output image
        """
        self.updater = Updater(token, use_context=True)
        self.job_queue = self.updater.job_queue

        self.handlers = collections.defaultdict(self.dialog)
        handler = MessageHandler(Filters.all, self.handle_message)
        self.updater.dispatcher.add_handler(handler)

        self.dict_running_captions = {}
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        self.cg = Caption_Net.CaptionGenerator('vocab.npy', 'best_model_1.pt')

    def start(self):
        """
        Start telegram bot pooling
        """
        self.updater.start_polling()

    def dialog(self):
        """
        Generator function, that return the answer depend on the input message
        """

        yield from self._request_command('/start', 'Type /start to begin')
        yield from self._request_command('/cc', "Hello, it's Captionizer bot for "
                                          "creation english captions to pictures, sent by you 😁."
                                          "To start photo analyzing and captions creation, type /cc (short for Create Captions). To terminate our dialog "
                                          "at any stage, type /quit 😥")

        chat_id, image = yield from self._request_image('beautiful')

        self.create_captions(image, chat_id)

    @staticmethod
    def _load_image(image):
        """
        Method for image loading
        ----------
        image : telegram.PhotoSize or telegram.Document object that represent picture
        """
        return image.get_file().download()

    @staticmethod
    def _request_command(command_name, first_message):
        """
        Generator function that will permanently ask user to send command_name command
        Parameters
        ----------
        command_name : name of the disired command
        first_message: first message that will be send and repeated until the 
        user enters 'command_name'
        """

        answer = yield first_message
        while answer.text != '{}'.format(command_name):
            answer = yield "Type {} command".format(command_name)

    @staticmethod
    def _request_image(image_request_type):
        """
        Generator function that will permanently ask user to send Picture
        Parameters
        ----------
        image_request_type : image type that will be send in message to user
        """
        answer = yield "I'm waiting for your {} image to create captions for".format(image_request_type)
        attachment = answer.effective_attachment
        chat_id = answer.chat_id

        if type(attachment) == list:
            attachment = attachment[-1]

        while not (type(attachment) == PhotoSize or type(attachment) == Document):
            answer = yield "I don't understand your query, send your {} image one more time".format(image_request_type)
            attachment = answer.effective_attachment
            if type(attachment) == list:
                attachment = attachment[-1]
        return chat_id, ImageHandlerBot._load_image(attachment)

    def handle_message(self, update, context):
        """
        Method that handles messages and send the response to user
        Parameters
        ----------
        update, context : parameters of user's message
        """
        chat_id = update.message.chat_id

        # In case of /quit command remove user's generator from handlers

        if update.message.text == "/quit":
            self.handlers.pop(chat_id)

        # If the user is not new send his message into generator

        if chat_id in self.handlers:
            try:
                answer = self.handlers[chat_id].send(update.message)
            except StopIteration:
                if chat_id in self.dict_running_captions:
                    answer = 'Your picture is processing'
                else:
                    answer = self.restart_dialog(chat_id)

        # In other cases the user is new, so lets create handler for him

        else:
            next(self.handlers[chat_id])
            answer = self.handlers[chat_id].send(update.message)

        # And finally send generator response to user

        context.bot.sendMessage(chat_id=chat_id, text=answer)

    def restart_dialog(self, chat_id):
        """
        Restart dialog with user
        In this case restart dialog, skip /start intro and 
        translate directly to the next step (waiting /cc command)
        Parameters
        ----------
        chat_id : number of chat id with particular user
        """
        self.handlers.pop(chat_id)
        next(self.handlers[chat_id])
        _ = self.handlers[chat_id].send(Message(*[0] * 4, text='/start'))  
        return self.handlers[chat_id].send(Message(*[0] * 4, text='dummy'))

    def create_captions(self, pic, chat_id):
        """
        Method to start creating captions to the image
        Parameters
        ----------
        pic_context: path to contex picture
        pic_style  : path to style picture
        chat_id    : identification number of chat with user
        """

        # Add future object into queue of thread executor
        #self.nst.run_style_transfer(pic_context, pic_style)
        future = self.executor.submit(
            self.cg.get_caption_message,
            pic
        )

        # Insert future to dictionary of currently running futures and
        # add repeating job to check if this future finish captions creation

        self.dict_running_captions[chat_id] = future
        #dict_final_pics[chat_id].exception(10)
        def check_image_ready(context, dict_final_pics, picture_rm, chat_id):
            job = context.job
            if dict_final_pics[chat_id].done():
                caption_result = dict_final_pics[chat_id].result()
                dict_final_pics.pop(chat_id)
                context.bot.sendMessage(chat_id=chat_id,
                                        text=caption_result)

                os.remove(pic)

                job.schedule_removal()

        self.job_queue.run_repeating(
            lambda context: check_image_ready(context,
                                              self.dict_running_captions,
                                              pic,
                                              chat_id), 5
        )

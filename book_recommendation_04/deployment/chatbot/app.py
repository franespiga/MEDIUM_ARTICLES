import bot_utils
import elastic_utils
import requests
import tensorflow as tf 

from wasabi import msg
import typer

from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    ConversationHandler,
    CallbackContext,
)

BOT_TOKEN = # YOUR BOT TOKEN

msg.info("Connecting to ElasticSearch...")
HOST = 'localhost'
PORT = 9200
# CONNECT TO INSTANCE
es = elastic_utils.get_connection_to_es(HOST, PORT)
msg.good(f"Connected to ElasticSearch node {HOST}:{PORT}")

# CREATE EMBEDDINGS
#from sentence_transformers import SentenceTransformer, util
#BASE_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'
#msg.info("Loading model")
#model = SentenceTransformer(BASE_MODEL)
#msg.good("Embedding model loaded")

embedding_servicer = "http://localhost:8502/embed_text"

def embed(query, server_url : str):
    print(f"Sending to url {server_url}")
    print(f"Description: {query}")
    r = requests.post(
        server_url, json = {'text':query}, timeout = 8000
    )
    candidates = r.json()
    embedding = tf.nn.l2_normalize(tf.constant(candidates['embedding'], shape = (1,384)), axis=1).numpy().tolist()[0]

    return embedding




def prettify_results(results):
    pretty_results = []

    cols = ['title', 'description', 'authors', 'categories', 'ISBN']
    for r in results:
        aux = {}
        for c in cols:
            aux[c] = r[c]
        pretty_results.append(aux)
    return pretty_results


def suggest(update: Update, context: CallbackContext) -> int:
    """Stores the selected gender and asks for a photo."""
    user = update.message.from_user
    description = update.message.text

    # Embed message
    msg.info("Sending the query to the embedding service...")
    description_embedding = embed(description, embedding_servicer)
    print(description_embedding)
    msg.good("Embedding computed", f"Dimension: {len(description_embedding)}")

    # RETRIEVING TOP K RESULTS BASED ON DESCRIPTION
    msg.info("Retrieving candidates based on description:", description)
    results = elastic_utils.get_most_similar(description_embedding, ['description_embedding', 'categories_embedding'], es, 'books', 10, timeout_secs = 300, debug_mode = True)
    print(prettify_results(results['results']))

    update.message.reply_text(
        '\n\n'.join([bot_utils.process_message(r) for r in prettify_results(results['results'])]),
        reply_markup=ReplyKeyboardRemove(),parse_mode='markdown'
    )

    return True

def cancel(update: Update, context: CallbackContext) -> int:
    """Cancels and ends the conversation."""
    user = update.message.from_user

    update.message.reply_text(
        'Bye! I hope we can talk again some day.', reply_markup=ReplyKeyboardRemove()
    )

    return ConversationHandler.END


def main(    
    bot_token : str = typer.Argument(BOT_TOKEN, help="Telegram BOT token")
    ):



    # START BOT POLLING
    msg.info("Connecting to bot...")
    updater = Updater(bot_token)
    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    dispatcher.add_handler(MessageHandler(Filters.text, suggest))
    # Add conversation handler with the states GENDER, PHOTO, LOCATION and BIO
    #conv_handler = ConversationHandler(
    #    entry_points=[CommandHandler('start', start)],
    #    states={
    #        SUGGEST: [MessageHandler(Filters.text, suggest)],
    #    },
    #    fallbacks=[CommandHandler('cancel', cancel)],
    #)
    #dispatcher.add_handler(conv_handler)

    # Start the Bot
    msg.good("Connection successful, starting polling...")
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()



if __name__ == '__main__':
    typer.run(main)
def telegram_bot_sendtext(bot_message, bot_chatID):
    bot_token = # YOUR BOT TOKEN
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message

    response = requests.get(send_text)
    return response.json() 

def process_message(result):
    message = [
        f"*{result['title']}*\n",
        f"Autores:{', '.join(result['authors'])}\n",
        f"ISBN:{result['ISBN']}\n",
        #f"resumen:{', '.join(result['description'])}\n",
    ]

    return ''.join(message)
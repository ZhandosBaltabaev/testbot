import openai
import logging
import json
from aiogram import Bot, Dispatcher, executor, types

# Включаем логирование
logging.basicConfig(level=logging.INFO)

# Загружаем конфиг
with open('C:/Users/DeadBeat/Documents/bot/test1/config.json', 'r') as file:
    config = json.load(file)

# API ключи
openai.api_key = config['openai']
TOKEN = config['token']

# Создаём бота и диспетчер
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

# История сообщений
messages = [
    {"role": "system", "content": "Вы — консультант по курсам TechLab. Вы помогаете пользователям с вопросами о курсах, расписании и стоимости."}
]

# Обработчик сообщений
@dp.message_handler()
async def send(message: types.Message):
    logging.info(f"Получено сообщение: {message.text}")
    
    messages.append({"role": "user", "content": message.text})  # Добавляем в историю

    try:
        response = openai.ChatCompletion.create(  # ✅ Используем актуальный метод
            model="ft:gpt-3.5-turbo-0125:personal::BSTegYN2",
            messages=messages
        )

        bot_reply = response["choices"][0]["message"]["content"]  # ✅ Новый способ извлечения ответа
        messages.append({"role": "assistant", "content": bot_reply})  # Добавляем ответ в историю

        await message.answer(bot_reply)

    except Exception as e:
        logging.error(f"Ошибка OpenAI: {e}")
        await message.answer("Ошибка при обработке запроса.")

# Запускаем бота
if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
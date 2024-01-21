from flask import Flask, request, jsonify
import aiosqlite
import asyncio
import os, json, re, numpy as np
from PIL import Image
from io import BytesIO
from tensorflow.keras.models import load_model
import aiohttp

app = Flask(__name__)

loaded_model = load_model('model.h5', compile=False)
with open('data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

async def preprocess_image(image):
    image = image.resize((64, 64))
    image = np.array(image)
    image1 = image / 255.0
    image = np.expand_dims(image1, axis=0)
    return image

async def connect_to_database():
    db = await aiosqlite.connect("pokemon.db")
    await db.execute("CREATE TABLE IF NOT EXISTS pokies (command str)")
    print("pokies table created!!")
    return db

@app.route('/predict', methods=['POST'])
def predict_pokemon():
    try:
        image_url = request.json['image_url']

        async def fetch_image():
            async with aiohttp.ClientSession() as session:
                async with session.get(url=image_url) as resp:
                    if resp.status == 200:
                        content = await resp.read()
                        return BytesIO(content)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        image_data = loop.run_until_complete(fetch_image())
        image = Image.open(image_data)

        preprocessed_image = loop.run_until_complete(preprocess_image(image))
        predictions = loaded_model.predict(preprocessed_image)
        data_x = np.argmax(predictions, axis=1)
        pokemon_name = list(data.keys())[data_x[0]]
        print(pokemon_name)
        
        return jsonify({"pokemon_name": pokemon_name})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    asyncio.run(connect_to_database())
    app.run(debug=True)

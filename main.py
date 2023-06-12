import os
import base64
from PIL import Image
import io

import tensorflow as tf
from tensorflow.keras.models import load_model

from pydantic import BaseModel
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

class ImageRequest(BaseModel):
  img: str

root = os.path.dirname(os.path.abspath(__file__))
app = FastAPI()

app.mount("/public", StaticFiles(directory="public"), name="public")
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

@app.get("/")
async def load_root():
  with open(os.path.join(root, 'public/index.html')) as fh:
    data = fh.read()
  return Response(content=data, media_type="text/html")

@app.post("/file")
async def create_file(req: ImageRequest):
  model = load_model('./models/model_15.h5')

  base64_img = req.img
  base64_img = base64_img.split(",")[1]

  img_bytes = base64.b64decode(base64_img)

  img = Image.open(io.BytesIO(img_bytes))

  # Imagen a tensor
  img_tensor = tf.keras.preprocessing.image.img_to_array(img)

  # Resize the image
  new_size = (180, 180)
  resized_img = tf.image.resize(img_tensor, new_size)

  img_array = tf.keras.utils.img_to_array(resized_img)
  img_array = tf.expand_dims(img_array, 0) # Create a batch

  predictions = model.predict(img_array)
  predictions = predictions.tolist()
  print(predictions[0][0])

  return {"prediction": predictions[0][0], "error": False}

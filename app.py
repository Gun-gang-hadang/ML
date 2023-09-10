import os
import uvicorn
import configparser
import torch.utils.data
from fastapi import FastAPI, UploadFile
import tensorflow as tf
from food_detection import detect_and_crop_image
from food_recognition import predict_food


app = FastAPI(max_request_size = 1024*1024*1024)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = tf.keras.models.load_model('./food_recognition_model2.h5')

dir_name = "crop"
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

config = configparser.ConfigParser()
config.read('config.ini')

host = config.get('server', 'host')
port = config.getint('server', 'port')


@app.get("/")
async def root():
    return {"message": f"Server running on {host}:{port}"}


@app.post("/api/v1/analyze/image")
async def preprocess_image(image: UploadFile):
    print("*")
    detect_and_crop_image(image, dir_name)
    food_name = predict_food()
    print(food_name)
    return food_name


if __name__ == '__main__':
    print(host)
    uvicorn.run(app, host=host, port=port)

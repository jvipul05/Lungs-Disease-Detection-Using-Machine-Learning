from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
app = FastAPI()
origins = ["*"]
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = load_model(r'D:\Lungs Xray\model.h5')



@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
        img = Image.open(BytesIO(data))

        # Convert the image to a NumPy array
        img_array = img_to_array(img)

        # Reshape the image to match the expected input shape of the model
        img_array = img_array.reshape((1,) + img_array.shape)

        # Normalize pixel values to be in the range [0, 1]
        img_array /= 255.0

        return img_array

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    
    
    predictions = MODEL.predict(image)
    class_label = "Pneumonia" if predictions[0][0] > 0.5 else "Normal"
    Probability = predictions[0][0]
    
    #predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    #confidence = np.max(predictions[0])
    return {
        'class': class_label,
        'probability': float(Probability)
     
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)


from fastapi import FastAPI, HTTPException, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import math

app = FastAPI()

# Allow all origins for CORS (you can customize this based on your needs)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model(f"saved_models/1")
CLASS_NAMES = ["Monkeypox", "Others"]

def read_file_as_image(data) -> np.ndarray:
    img = Image.open(BytesIO(data)).convert("RGB")
    img = img.resize((224, 224))
    image = np.array(img)
    return image

@app.post("/predict")
async def predict(image: UploadFile = Form(...)):
    try:
        image_data = await image.read()
        image_np = read_file_as_image(image_data)
        img_batch = np.expand_dims(image_np, 0)
        predictions = model.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = math.floor(float(np.max(predictions[0]))*100)

        return {"class": predicted_class, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='localhost', port=8009)
from fastapi import FastAPI
from schemas import ImageData
from model import load_model, preprocess_image, predict

# Initialize the FastAPI app
app = FastAPI()

# Load the model
model, device = load_model()

@app.post("/classify")
async def classify(image_data: ImageData):
    """Endpoint for classifying an image."""
    input_tensor = preprocess_image(image_data.image, device)
    prediction, probabilities = predict(input_tensor, model, device)
    return {"prediction": prediction, "probabilities": probabilities}


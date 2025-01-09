import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from transformers import AutoModelForImageClassification
import os

# get the absolute path the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# define the path to the local model directory
LOCAL_MODEL_PATH = os.path.join(current_dir, "models")

# Define constants
MODEL_NAME = "farleyknight/mnist-digit-classification-2022-09-04"
# Jos sovellus käynnistetään ilman Docker-ympäristöä ylempänä olevasta kansiosta
# muodostuu "models"-kansio sen alle, ja malli tallennetaan sinne. Jos taas sovellus
# käynnistetään samasta kansiosta, jossa tämä tiedosto on, malli tallennetaan suoraan
# "models"-kansioon. Tämä on haluttu toimintatapa, jos sovelluksesta halutaan tehdä
# Docker-kontti, niin tällöin malli on helmpompi tallentaa osaksi konttia.
# Tällöin mallia ei tarvitse ladata joka kerta, kun kontti käynnistetään. Tämä nopeuttaa
# sovelluksen käynnistymistä ja vähentää mallin lataamiseen kuluvaa aikaa. Lisäksi jos
# palvelu pyörii pilvessä, mallin lataaminen joka kerta voi aiheuttaa ylimääräisiä kustannuksia.
# dot-env:iä voisi käyttää, jos haluttaisiin määrittää mallin sijainti ympäristömuuttujana.

IMAGE_SIZE = (224, 224)
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

# Load the Hugging Face model
def load_model():
    try:
        # Load the model from the local directory if it exists
        model = AutoModelForImageClassification.from_pretrained(LOCAL_MODEL_PATH)
    except:
        # Load the model from the Hugging Face model hub if it does not exist locally
        # and save it to the local directory
        model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
        model.save_pretrained(LOCAL_MODEL_PATH)
    
    model.eval()  # Set the model to evaluation mode
    # Determine the device to run the model on (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    return model, device

# Preprocessing pipeline for the image
def preprocess_image(image_data: list, device: torch.device) -> torch.Tensor:
    """
    Preprocesses the input image data for model classification.

    This function converts the input image data (a list of pixel values) to a PyTorch tensor
    suitable for model input. The image is processed with the following transformations:
        1. Resized to the target size (224x224).
        2. Converted from a NumPy array to a PIL image.
        3. Normalized using a mean of [0.5, 0.5, 0.5] and std of [0.5, 0.5, 0.5] for each RGB channel.
        4. Converted to a PyTorch tensor.
        5. A batch dimension is added to the tensor for model input.

    Args:
        image_data (list): A 3D list representing the image with RGB channels in the format [height, width, 3].

    Returns:
        torch.Tensor: The preprocessed image as a PyTorch tensor with shape (1, 3, 224, 224), ready for classification.
    """
        
    image = np.array(image_data, dtype=np.uint8)
    pil_image = Image.fromarray(image)

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),  # Resize to IMAGE_SIZE
        transforms.ToTensor(),  # Convert to PyTorch tensor
        transforms.Normalize(mean=MEAN, std=STD)  # Normalize for 3 channels
    ])

    input_tensor = transform(pil_image).unsqueeze(0)
    return input_tensor.to(device)

# Perform model inference
def predict(input_tensor: torch.Tensor, model: torch.nn.Module, device: torch.device):
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(input_tensor)
    
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)[0].tolist()
    prediction = torch.argmax(logits, dim=1).item()

    return prediction, probabilities

from pydantic import BaseModel, field_validator
from typing import List

class ImageData(BaseModel):
    image: List[List[List[int]]]  # List of RGB pixel values in a 3D list format

    @field_validator("image")
    @classmethod
    def validate_image_dimensions(cls, image):
        if not isinstance(image, list) or len(image) == 0:
            raise ValueError("Image must be a non-empty list.")
        if not all(isinstance(row, list) for row in image):
            raise ValueError("Image must be a 2D list (list of lists).")
        if not all(isinstance(pixel, list) and len(pixel) == 3 for row in image for pixel in row):
            raise ValueError("Each pixel must be a list with 3 values (RGB).")
        return image

import requests
import os

class BackendAPI:
    def __init__(self, url=None):
        self.url = url or os.getenv("ENDPOINT_URL", "http://127.0.0.1:8080/classify")

    def call_backend(self, image):
        """Calls the backend to classify the image."""
        try:
            response = requests.post(
                self.url,  # ENDPOINT URL
                json={"image": image.tolist()},
            )
            if response.status_code == 200:
                return response.json()
            else:
                raise ValueError(f"Backend returned status code {response.status_code}")
        except Exception as e:
            raise ValueError(f"Error communicating with the backend {self.url}: {e}")

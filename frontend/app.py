import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from backend_api import BackendAPI 
import os

### Backend API-url täytyy muuttaa sen mukaan onko käytössä paikallinen vai Docker-ympäristö
### Docker-ympäristössä sovellukset kommunikoivat toistensa kanssa palvelunimien avulla
### Jos kontit on käynnistetty erikseen, toimivat ne omissa erillisissä Docker-verkkoavaruuksissaan
### eivätkä löydä toisiaan. 
### Kun sovellukset käynnistetään yhdessä Docker Compose -tiedostolla, ne ovat samassa verkkoavaruudessa.
### Normaalisti sovellukset kommunikoivat toistensa kanssa localhost-osoitteella ja porttinumerolla 

# Initialize the backend API without Docker, default URL is http://127.0.0.1:8000/classify
#backend_api = BackendAPI()
# Initialize the backend API with Docker Compose serice name, default URL is http://fastapi:8000/classify
#backend_api = BackendAPI(url="http://fastapi:8000/classify")
# Initialize the backend API with Docker by exposing the FastAPI service to the host machine 

#port = int(os.environ.get("PORT", 8080))

backend_api = BackendAPI() 

# Function to initialize session state
def initialize_session_state():
    if "probabilities" not in st.session_state:
        st.session_state.probabilities = [0.0] * 10
    if "prediction" not in st.session_state:
        st.session_state.prediction = None
    if "success_message" not in st.session_state:
        st.session_state.success_message = None  # Initialize to None

# Function to display top 3 most probable digits
def display_top_3_probabilities():
    st.write("### Top 3 Probabilities:")
    if st.session_state.probabilities:
        # Get indices of the top 3 most probable digits
        top_3_indices = sorted(range(len(st.session_state.probabilities)),
                               key=lambda i: st.session_state.probabilities[i], reverse=True)[:3]
        
        # Display the top 3 digits with their probabilities
        for index in top_3_indices:
            st.write(f"Digit {index}: {st.session_state.probabilities[index]:.2f}")

# Function to handle the classification button click
def handle_classification(canvas_result):
    # Check if the canvas has any drawn content
    if canvas_result.image_data is None or np.sum(canvas_result.image_data[:, :, :3]) == 0:
        st.warning("Please draw a digit on the canvas before classifying.") 
    else:
        # Retrieve the drawn image directly in RGB format
        image_data = canvas_result.image_data[:, :, :3]  # Retain RGB channels
        image_data = 255 - image_data  # Invert colors to match MNIST style
        image_data = image_data.astype(np.uint8)  # Ensure uint8 type

        # Call the backend for classification
        result = backend_api.call_backend(image_data)

        if result:
            st.session_state.probabilities = result['probabilities']
            st.session_state.prediction = result['prediction']
            st.session_state.success_message = f"The model predicts: {result['prediction']}"  # Store the success message
            # Trigger a rerun to immediately reflect updated session state
            st.rerun()

# Function to handle the clear button
def handle_clear():
    st.session_state.clear_count = st.session_state.get('clear_count', 0) + 1
    st.session_state.probabilities = [0.0] * 10
    st.session_state.prediction = None
    st.session_state.success_message = None  # Clear the success message
    st.rerun()

# Main function
def main():

    st.title("Digit Drawing App")
    st.write("Draw a number 0-9 on the canvas below using your mouse.")

    # Unique key for the canvas to handle clearing
    canvas_key = f"canvas_{st.session_state.get('clear_count', 0)}"

    # Drawing canvas configuration
    canvas_result = st_canvas(
        fill_color="#000000",  # Black pen
        stroke_width=10,
        stroke_color="#FFFFFF",  # White strokes
        background_color="#000000",  # Black background
        height=280,
        width=280,
        drawing_mode="freedraw",
        key=canvas_key,
    )

    # Initialize session state
    initialize_session_state()

    # Display top 3 most probable digits
    display_top_3_probabilities()

    # Display the success message if available
    if st.session_state.success_message:
        st.success(st.session_state.success_message)

    # Classify button
    if st.button("Classify"):
        handle_classification(canvas_result)

    # Clear button
    if st.button("Clear"):
        handle_clear()

if __name__ == "__main__":
    main()

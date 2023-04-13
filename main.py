import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import pytesseract
import numpy as np
from train_g import recogniser01
def navigate_pages():
    pages = {
        "Home": homepage,
        "0 and 1 recognition": recogniser01,
    }
    session_state = st.session_state.setdefault("page", "Home")

    # Create a selectbox to choose the page.
    page_choice = st.sidebar.selectbox("Select a page", options=list(pages.keys()))

    # If the page choice is different than the current page, update the session state.
    if page_choice != session_state:
        session_state = page_choice

    # Call the function that renders the current page.
    pages[session_state]()

def homepage():
    canvas_width = 400
    canvas_height = 400
    canvas_result = st.empty()
    st.header("OCR writing recognition")
    st.warning("""It is difficult for the model to recognize a single digit or letter, such as "1" or "a". It is suggested to write words inside the canvas instead.""")
    # Create a canvas for drawing
    canvas = st_canvas(
        fill_color="#ffffff",
        stroke_width=8,
        stroke_color="#000000",
        background_color="#ffffff",
        width=canvas_width,
        height=canvas_height,
        drawing_mode="freedraw",
        key="canvas",
    )
    if st.button("Load and OCR"):
        saved_image = canvas.image_data.astype(np.uint8)
        image = Image.fromarray(saved_image)
        #print(saved_image)
        text = pytesseract.image_to_string(image)
        st.write("OCR Result:")
        st.write(text)

navigate_pages()
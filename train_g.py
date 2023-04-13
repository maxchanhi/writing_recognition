from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from nn import *
def recogniser01():
    canvas_width = 400
    canvas_height = 400
    canvas_result = st.empty()
    st.header("OCR writing recognition")
    st.warning("""Write "0" or "1"! <br>
    This model was specifically trained for recognising 0 and 1.""")
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

    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    #image = Image.open("training_data/t2.png").convert("RGB")
    if st.button("Load and OCR") == True:
        saved_image = canvas.image_data.astype(np.uint8)
        image = Image.fromarray(saved_image)
        image = image.convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        # Print prediction and confidence score
        print(index)
        st.success(f"{class_name}")
        st.success(f"Confidence Score: {confidence_score}")
    else:
        pass
if __name__ == "__main__":
    recogniser01

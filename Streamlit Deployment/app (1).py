import streamlit as st
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import easyocr
import numpy as np

st.set_page_config(page_title="Number Plate Detector & OCR", page_icon="🚗")

st.title("Traffic Management System !")
st.write("Please Upload the Image")
st.markdown("---")

@st.cache_resource
def load_yolo_model():
    model = YOLO('best (2).pt')  # Make sure this file exists in the directory
    return model

model = load_yolo_model()

@st.cache_resource
def load_ocr_reader():
    reader = easyocr.Reader(['en'])
    return reader

reader = load_ocr_reader()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    Img = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)

    st.subheader("Original Image")
    st.image(image_rgb, caption="Uploaded Image", use_column_width=True)
    st.markdown("---")

    st.subheader("YOLO Detection Result")
    results = model(image_rgb)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    annotated_frame = results[0].plot()
    ax.imshow(annotated_frame)
    ax.axis('off')
    st.pyplot(fig)

    plate_found = False
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            crop_mask = image_rgb[y1:y2, x1:x2]
            if crop_mask.shape[0] > 0 and crop_mask.shape[1] > 0:
                plate_found = True
                st.subheader(f"Plate {i+1}")
                st.image(crop_mask, caption=f"Cropped Plate {i+1}")
                OCR_Pic = reader.readtext(crop_mask)
                if OCR_Pic:
                    for detection in OCR_Pic:
                        text = detection[1]
                        st.success(f"Detected Text: {text}")
                else:
                    st.info("No text found.")
    if not plate_found:
        st.warning("No plates detected.")

!pip install ultralytics
!pip install opencv-python
!pip install matplotlib
!pip install pandas
!pip install easyocr
# Importing the Dependencies:-

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt 

# Loading the yolo model:-
model = YOLO('/content/best (1).pt')

# Taking the image and importing for the processing:-

image = '/content/CAR2.jpg'

# Read the image:-
Img = cv2.imread(image)

# Taking the Color code (RGB):-
image_rgb = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)

# Run YOLO MODEL:-
results = model(image_rgb)

# Outlining the model:-
results[0].plot()  
plt.imshow(results[0].plot())
plt.axis('off')
plt.show()

# Taking the Plate into the separate:-

for result in results:
    boxes = result.boxes.xyxy.cpu().numpy().astype(int)  # xyxy format
    for box in boxes:
        x1, y1, x2, y2 = box
        crop_mask = image_rgb[y1:y2, x1:x2]

        # Crpped image:-
        plt.imshow(crop_mask)
        plt.title("Number plate Of Car")
        plt.axis('off')
        plt.show()

import easyocr

# Initialize OCR reader:-
reader = easyocr.Reader(['en'])

# Perform OCR on the cropped plate
OCR_Pic = reader.readtext(crop_mask)

# Display OCR result
for detection in OCR_Pic:
    text = detection[1]
    print("Detected Text:", text)
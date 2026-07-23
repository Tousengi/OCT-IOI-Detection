from ultralytics import YOLO
import os
import numpy as np

image_foler = "testset/images"
test_model_path = "test_model_path/best.pt"

model = YOLO(test_model_path)

for img in os.listdir(image_folder):
  img_path = os.path.join(image_folder, img)
  results = model(img_path)

# %%
# yolo 불러오기
from ultralytics import YOLO
import cv2
import streamlit as st

# %%
# %pip install ultraLytics

# %%
yolo = YOLO("yolo11x.pt")

# %%
img = cv2.imread('zidane.jpg')
yolo.predict(img, save=True, imgsz=640, conf=0.5)


# %%

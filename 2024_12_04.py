# %%
from ultralytics import YOLO
import cv2

# %%
model = YOLO('yolo11n.pt')
model.info()

# %%
img = cv2.imread('bus.jpg')
model.predict(img, save=True, imgsz=640, conf=0.5)

# %%
results = model(img)

for result in results:
    print(result)

# %%
result = results[0]
result_img = img.copy()

for box in result.boxes:
    x1, y1, x2, y2 = box.xyxy[0]
    cv2.rectangle(result_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    print(int(box.cls.cpu().item()))
    print(box.conf.cpu().item())

# %%
display(Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)))
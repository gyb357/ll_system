# %%
from ultralytics import YOLO
import cv2
from PIL import Image

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
    print(int(box.cls.cpu().item()))
    print(box.conf.cpu().item())

    _class = int(box.cls.cpu().item())
    if _class == 5:
        cv2.rectangle(result_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    else:
        cv2.rectangle(result_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

# %%
Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)).show()

# %%
model = YOLO('yolo11n-seg.pt')
model.info()

# %%
result = model(img)[0]
result_img = img.copy()

# %%
print(result.masks)

# %%
import numpy as np

for mask_data in result.masks.data:
    mask_img = mask_data.cpu().numpy()
    mask_img = np.where(mask_img > 0.5, 255, 0).astype(np.uint8)

# %%
Image.fromarray(cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)).show()

# %%
img_h, img_w, _ = img.shape

for _segment in result.masks.xyn:
    np_cnt = (_segment*(img_w, img_h)).astype(np.int32)
    cv2.polylines(result_img, [np_cnt], isClosed=True, color=(0, 255, 0), thickness=2)

# %%
Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)).show()

# %%

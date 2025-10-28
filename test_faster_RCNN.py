import torch
from torchvision.transforms import ToTensor
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt

# --- 1. Load model ---
num_classes = 21  # VOC có 20 lớp + background
model = fasterrcnn_mobilenet_v3_large_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load("fasterrcnn_mobilenet_weights.pth", map_location="cpu"))
model.eval()

# --- 2. Load ảnh test ---
img = Image.open("test_image.jpg").convert("RGB")
transform = ToTensor()
img_tensor = transform(img).unsqueeze(0)  # (1, C, H, W)

# --- 3. Dự đoán ---
with torch.no_grad():
    predictions = model(img_tensor)

# --- 4. Xem kết quả ---
print(predictions)
# predictions[0]['boxes']  → toạ độ các khung
# predictions[0]['labels'] → nhãn dự đoán
# predictions[0]['scores'] → độ tin cậy

# --- 5. (Tuỳ chọn) Hiển thị hình ---
boxes = predictions[0]['boxes']
plt.imshow(img)
for box in boxes:
    x1, y1, x2, y2 = box
    plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                      fill=False, color='red', linewidth=2))
plt.show()
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

# --- 2. Định nghĩa class names cho VOC dataset ---
VOC_CLASSES = [
    '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# --- 3. Load ảnh test ---
img = Image.open("download.jpg").convert("RGB")
transform = ToTensor()
img_tensor = transform(img).unsqueeze(0)  # (1, C, H, W)

# --- 4. Dự đoán ---
with torch.no_grad():
    predictions = model(img_tensor)

# --- 5. Lọc kết quả với ngưỡng confidence ---
confidence_threshold = 0.5
boxes = predictions[0]['boxes']
scores = predictions[0]['scores']
labels = predictions[0]['labels']

# Lọc các detection có confidence >= ngưỡng
keep = scores >= confidence_threshold
filtered_boxes = boxes[keep]
filtered_scores = scores[keep]
filtered_labels = labels[keep]

print(f"Found {len(filtered_boxes)} detections with confidence >= {confidence_threshold}")

# --- 6. Hiển thị hình với bounding boxes và nhãn ---
fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(img)

for i, (box, score, label) in enumerate(zip(filtered_boxes, filtered_scores, filtered_labels)):
    x1, y1, x2, y2 = box
    
    # Vẽ bounding box
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                         fill=False, color='red', linewidth=2)
    ax.add_patch(rect)
    
    # Tạo nhãn với tên class và confidence
    class_name = VOC_CLASSES[label]
    label_text = f'{class_name}: {score:.3f}'
    
    # Thêm nhãn ở góc trên trái của bounding box
    ax.text(x1, y1 - 5, label_text, 
            fontsize=10, color='white', weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.8))

plt.title(f'Object Detection Results - {len(filtered_boxes)} objects detected')
plt.axis('off')
plt.tight_layout()
plt.show()

# --- 7. In kết quả chi tiết ra console ---
print("\n--- Detailed Detection Results ---")
for i, (box, score, label) in enumerate(zip(filtered_boxes, filtered_scores, filtered_labels)):
    class_name = VOC_CLASSES[label]
    print(f"Object {i+1}: {class_name} (confidence: {score:.3f})")
    print(f"  Bounding box: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
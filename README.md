# 🐾 Animal Detection with Faster R-CNN

Deep Learning project for detecting animals in images using the **Faster R-CNN** object detection architecture trained on the **PASCAL VOC dataset**.

---

# 📖 Overview

This project implements an **object detection system** capable of recognizing animals in images using **Faster R-CNN**, a two-stage deep learning model known for its high detection accuracy.

The system detects animals and predicts:

* 📦 Bounding boxes
* 🏷 Object classes
* 📊 Confidence scores

The model is trained using the **PASCAL VOC dataset**, focusing on several animal categories.

---

# 🐶 Animal Classes

The model detects the following animals:

* Dog
* Cat
* Horse
* Cow
* Sheep

These classes are extracted from the **PASCAL VOC dataset**.

---

# 🎥 Demo Video

Click the image below to watch the demo video.

Video befor:

https://github.com/user-attachments/assets/6a8ddeb6-4a0f-495b-945a-7387a8cc61ee

Video after:

---

# 🖼 Example Detection

Example output of the model:

```
Dog   | Confidence: 0.94
Cow   | Confidence: 0.89
```

Bounding boxes are drawn around detected animals.

Example visualization:

```
+-------------------------+
|                         |
|        🐶 Dog           |
|     [ Bounding Box ]    |
|                         |
+-------------------------+
```

---

# 🧠 Model Architecture

Faster R-CNN is a **two-stage object detection model**.

### 1️⃣ Feature Extraction

A convolutional neural network extracts **feature maps** from the input image.

### 2️⃣ Region Proposal Network (RPN)

The RPN proposes **candidate object regions** that might contain objects.

### 3️⃣ Detection Head

Each region proposal is processed to:

* classify object category
* refine bounding box coordinates

This architecture allows **accurate localization and classification**.

---

# 📂 Dataset

This project uses the **PASCAL VOC dataset**, a widely used benchmark for object detection.

Dataset includes:

* Annotated images
* Bounding boxes
* XML annotation files

Example dataset structure:

```
VOCdataset
 ├── JPEGImages
 ├── Annotations
 ├── ImageSets
 └── SegmentationClass
```

Annotation example:

```
<object>
    <name>dog</name>
    <bndbox>
        <xmin>48</xmin>
        <ymin>240</ymin>
        <xmax>195</xmax>
        <ymax>371</ymax>
    </bndbox>
</object>
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/faster-rcnn-animal-detection.git
cd faster-rcnn-animal-detection
```

Install dependencies:

```bash
pip install torch torchvision
pip install numpy matplotlib opencv-python
```

---

# 🚀 Training

Train the Faster R-CNN model:

```bash
python train_faster_RCNN.py
```

Training pipeline:

1. Load VOC dataset
2. Parse XML annotations
3. Generate region proposals
4. Train Faster R-CNN
5. Evaluate performance

---

# 🔍 Inference

Run detection on a test image:

```bash
python test_faster_RCNN.py
```

Output:

* bounding boxes
* predicted animal class
* confidence score

---

# 📁 Project Structure

```
faster-rcnn-animal-detection
│
├── fasterrcnn_mobilenet_weight.pth # checkpoin
│
│
├── test_faster_RCNN.py
├── train_faster_RCNN.py
|
│
└── README.md
```

---

# 📊 Technologies Used

* Python
* PyTorch
* Computer Vision
* Deep Learning
* Faster R-CNN

---

# 🎯 Applications

This project can be applied to:

* Wildlife monitoring
* Smart farming
* Image understanding
* AI research
* Automated surveillance systems

---

# 👨‍💻 Author

**AI & Deep Learning Explorer**

Interested in:

* Artificial Intelligence
* Machine Learning
* Deep Learning
* Computer Vision
* Building AI models from scratch

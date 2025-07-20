

---

```markdown
# 🫁 Chest X-Ray Multi-Label Classification Pipeline

A deep learning pipeline for **multi-label classification** of chest X-ray images, integrating **modern CNN architectures** with **traditional computer vision features** for improved interpretability and performance evaluation.

---

## 📑 Table of Contents

1. [Overview](#overview)  
2. [Dataset](#dataset)  
3. [Installation](#installation)  
4. [Project Structure](#project-structure)  
5. [Methodology](#methodology)  
6. [Results](#results)  
7. [Visualizations](#visualizations)  
8. [Usage Guide](#usage-guide)  
9. [Future Work](#future-work)  
10. [License](#license)  
11. [Author](#author)

---

## 🔍 Overview

Chest X-rays often reveal **multiple pathologies in a single image**, making this a **multi-label classification** problem. This project builds a robust pipeline that includes:

- Preprocessing and augmentation tailored for medical images  
- Deep learning-based feature extraction using **ResNet50**  
- Hybrid model combining CNN features with **ORB/SIFT** descriptors  
- Evaluation metrics for clinical insight  
- **Grad-CAM** for model interpretability

---

## 🗂 Dataset

We use the [NIH Chest X-ray Dataset](https://www.nih.gov/) consisting of:

- 112,120 frontal-view chest X-ray images  
- 15 disease labels (e.g., Atelectasis, Cardiomegaly, Effusion)  
- Multi-label annotations per image

📁 Expected Directory Structure:

```

project\_root/
├── xray\_images/          # All chest X-ray images
└── Ground\_Truth.csv      # Image-label mappings

````

---

## ⚙️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/chest-xray-classification.git
cd chest-xray-classification
````

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

✅ Key packages:

* `tensorflow` (2.x)
* `opencv-python`
* `scikit-learn`
* `pandas`, `numpy`
* `matplotlib`, `seaborn`

---

## 🧱 Project Structure

```
chest-xray-classification/
├── xray_images/                  # X-ray images
├── Ground_Truth.csv              # Original labels
├── train_metadata.csv            # Training split
├── val_metadata.csv              # Validation split
├── test_metadata.csv             # Testing split
├── best_resnet_model.h5          # Saved ResNet model
├── best_hybrid_model.h5          # Saved hybrid model
├── chest_xray_classification.py  # Main script
└── README.md                     # Project documentation
```

---

## 🧠 Methodology

### 🔄 1. Preprocessing & Augmentation

Images are resized to **224x224** and normalized. Augmentation strategies include:

* Rotation (±15°)
* Translation (up to 10%)
* Zoom (90–110%)
* Brightness/contrast shifts
* Gaussian noise
* CLAHE (Contrast Limited Adaptive Histogram Equalization)

---

### 📈 2. Feature Extraction

#### a. Deep Learning Features

* **ResNet50** (pretrained on ImageNet) as backbone
* Global average pooling for feature reduction

#### b. Traditional Features

* **ORB** (Oriented FAST and Rotated BRIEF)
* **SIFT** (Scale-Invariant Feature Transform)

---

### 🏗 3. Model Architectures

#### ✅ Standard ResNet Model

* ResNet50 (partially frozen)
* GlobalAveragePooling + Dense(1024, ReLU) + Dropout(0.5)
* Final Dense(15, Sigmoid) for multi-label output

#### 🔀 Hybrid Model

* Dual-input architecture:

  * ResNet50 feature extractor
  * ORB features processed via Dense layers
* Merged with concatenation layer
* Final sigmoid output layer

---

### ⚙ 4. Training Config

* **Optimizer**: Adam (lr=0.0001)
* **Loss**: Binary Cross-Entropy
* **Metrics**: Accuracy, AUC, Precision, Recall
* **Callbacks**:

  * EarlyStopping
  * ModelCheckpoint
  * ReduceLROnPlateau

---

## 📊 Results

### ✅ Evaluation Summary

| Metric    | ResNet Model | Hybrid Model |
| --------- | ------------ | ------------ |
| Micro AUC | 0.8284       | 0.8167       |
| Macro AUC | 0.6049       | 0.5836       |

📌 *The ResNet-only model performs slightly better, but the hybrid model improves interpretability.*

### 🩺 Class-wise AUC

Detailed class-level ROC curves are provided to identify model performance across diseases.

---

## 🖼 Visualizations

The pipeline supports multiple visual analysis tools:

1. **ORB/SIFT Keypoints** – Feature maps overlaid on X-rays
2. **Training Plots** – Accuracy, loss, and AUC progression
3. **Confusion Matrix** – Multilabel evaluation
4. **Grad-CAM** – Highlights X-ray regions that influenced predictions

---

## 🚀 Usage Guide

### 🔧 Train the Models

```bash
python chest_xray_classification.py
```

### 📦 Load Pretrained Models

```python
from tensorflow.keras.models import load_model

resnet_model = load_model('best_resnet_model.h5')
hybrid_model = load_model('best_hybrid_model.h5')

preds = resnet_model.predict(test_images)
```

### 📈 Evaluate & Visualize

Use the provided utilities to:

* Generate evaluation reports
* Plot ROC curves and confusion matrices
* Visualize Grad-CAM results

---

## 🔭 Future Work

1. Integrate **DenseNet** or **EfficientNet** backbones
2. Incorporate **patient metadata** for better clinical insights
3. Use advanced **feature fusion techniques** (e.g., attention)
4. Add **DICOM support** for hospital compatibility
5. Build a **web dashboard** for radiologist interaction

---

## 👨‍💻 Author

**Jitendra Kumar Gupta**
📧 [jitendraguptaaur@gmail.com](mailto:jitendraguptaaur@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/jitendra-kumar-30a78216a/)
🎓 M.Tech – IIT Kanpur | B.Tech – NIT Surat
🔍 Interests: Deep Learning, Medical Imaging, Generative AI

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

```

### ✅ Key Improvements:
- Better sectioning and use of icons/emojis for readability.
- Clarified technical language (e.g., what ORB/SIFT are doing).
- Made future work more specific and strategic.
- Modular formatting makes it GitHub-friendly and easy to read.

```

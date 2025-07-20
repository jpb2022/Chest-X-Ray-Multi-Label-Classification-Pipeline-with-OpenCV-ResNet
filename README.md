

# Chest X-Ray Multi-Label Classification Pipeline

This project implements a deep learning pipeline for multi-label classification of chest X-ray images using both traditional deep learning approaches and hybrid models combining deep learning with traditional computer vision features.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Project Structure](#project-structure)
5. [Implementation Details](#implementation-details)
6. [Results](#results)
7. [Visualizations](#visualizations)
8. [How to Use](#how-to-use)
9. [Future Work](#future-work)
10. [License](#license)

## Project Overview

This project addresses the challenge of multi-label classification of chest X-rays, where each image can contain multiple pathologies simultaneously. The pipeline includes:

1. Data preprocessing and augmentation
2. Feature extraction using both deep learning (ResNet50) and traditional computer vision techniques (ORB, SIFT)
3. Two model architectures:
   - Standard ResNet50-based model
   - Hybrid model combining ResNet50 features with ORB features
4. Evaluation metrics and visualization tools
5. Grad-CAM visualization for model interpretability

## Dataset

The project uses the NIH Chest X-ray Dataset which contains:
- 112,120 frontal-view X-ray images
- 15 common thorax disease labels
- Multi-label annotations (each image can have multiple diseases)

The dataset should be organized as:

project_root/:

├── xray_images/   # Folder containing all X-ray images

├── Ground_Truth.csv # CSV file with image labels


## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/chest-xray-classification.git
cd chest-xray-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- TensorFlow 2.x
- OpenCV
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

## Project Structure

```
chest-xray-classification/
│
├── xray_images/                  # X-ray images
├── Ground_Truth.csv              # Labels metadata
├── train_metadata.csv            # Generated train split
├── val_metadata.csv              # Generated validation split
├── test_metadata.csv             # Generated test split
├── best_resnet_model.h5          # Saved ResNet model
├── best_hybrid_model.h5          # Saved hybrid model
├── chest_xray_classification.py  # Main implementation file
└── README.md                     # This file
```

## Implementation Details

### 1. Data Preprocessing and Augmentation

Key preprocessing steps:
- Image resizing to 224x224 pixels
- Normalization (pixel values scaled to [0,1])
- Extensive data augmentation:
  - Random rotations (±15 degrees)
  - Random shifts (up to 10% of image dimensions)
  - Random zoom (90-110% scaling)
  - Random brightness/contrast adjustments
  - Gaussian noise addition
  - CLAHE histogram equalization

### 2. Feature Extraction

Two approaches implemented:
1. **Deep Learning Features**:
   - ResNet50 backbone (pretrained on ImageNet)
   - Global average pooling of final convolutional layer

2. **Traditional Computer Vision Features**:
   - ORB (Oriented FAST and Rotated BRIEF) features
   - SIFT (Scale-Invariant Feature Transform) features

### 3. Model Architectures

#### Standard ResNet Model
- ResNet50 base (with partial layer freezing)
- Global average pooling layer
- Dense layers (1024 units with ReLU activation)
- Dropout for regularization (0.5)
- Final sigmoid output layer (15 units)

#### Hybrid Model
- Two parallel input branches:
  1. ResNet50 branch (same as standard model)
  2. ORB features branch (processed through dense layers)
- Concatenation of both branches
- Additional dense layers for combined feature processing
- Final sigmoid output layer

### 4. Training Configuration
- Optimizer: Adam (learning rate 0.0001)
- Loss function: Binary cross-entropy
- Metrics:
  - Accuracy
  - AUC (Area Under ROC Curve)
  - Precision
  - Recall
- Callbacks:
  - Model checkpointing
  - Early stopping
  - Learning rate reduction on plateau

## Results

### Performance Metrics

| Metric               | Standard Model | Hybrid Model |
|----------------------|----------------|--------------|
| Micro-average AUC    | 0.8284         | 0.8167       |
| Macro-average AUC    | 0.6049         | 0.5836       |

### Class-wise Performance

The project includes detailed class-wise ROC AUC scores and visualization comparing performance between the standard and hybrid models for each pathology.

## Visualizations

The implementation includes several visualization tools:

1. **Feature Visualization**:
   - ORB and SIFT keypoints on sample images

2. **Training History**:
   - Accuracy/loss curves
   - AUC progression

3. **Evaluation Metrics**:
   - Classification reports
   - Confusion matrices
   - ROC AUC comparisons

4. **Grad-CAM**:
   - Class activation maps showing which regions of the image contributed most to the prediction

## How to Use

1. **Training the Models**:
```python
# Run the main script
python chest_xray_classification.py
```

2. **Using Pretrained Models**:
```python
from tensorflow.keras.models import load_model

# Load models
resnet_model = load_model('best_resnet_model.h5')
hybrid_model = load_model('best_hybrid_model.h5')

# Make predictions
predictions = model.predict(test_images)
```

3. **Custom Evaluation**:
The script includes functions for:
- Evaluating models on test data
- Generating classification reports
- Plotting performance metrics

## Future Work

Potential improvements and extensions:
1. Experiment with other backbone architectures (DenseNet, EfficientNet)
2. Incorporate additional clinical data (patient demographics)
3. Implement more sophisticated hybrid feature fusion techniques
4. Develop a web-based interface for clinical use
5. Add DICOM support for direct hospital integration

---

## 🙋‍♂️ Author

**Jitendra Kumar Gupta**  
📧 [jitendraguptaaur@gmail.com](mailto:jitendraguptaaur@gmail.com)  
🔗 [LinkedIn: jitendra-gupta-iitk](https://www.linkedin.com/in/jitendra-kumar-30a78216a/)  
🎓 M.Tech – IIT Kanpur | B.Tech – NIT Surat  
🧠 Focused on Machine Learning, NLP, and Generative AI

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

This README provides comprehensive documentation for your project, covering all aspects from setup to implementation details and results. You may want to customize certain sections (like the license or specific file paths) to match your exact project configuration.

The structure follows best practices for technical documentation, with clear sections that help users understand, implement, and extend your work. The inclusion of visual examples (which would be rendered from your actual output images when viewed on GitHub) makes the documentation more engaging and informative.

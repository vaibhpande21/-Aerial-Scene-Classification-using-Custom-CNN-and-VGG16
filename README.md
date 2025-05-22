# Aerial Scene Classification using Custom CNN and VGG16

This project classifies aerial imagery into 21 land-use categories using two deep learning approaches: a custom-built Convolutional Neural Network (CNN) and a fine-tuned VGG16 transfer learning model. It demonstrates data preprocessing, augmentation, model training, evaluation, and performance visualization. Fine-tuned VGG16 achieved the highest accuracy at 93%.

---

## 1. Data Management

### 1.1 Experimental Protocol
We used an **80/20 train-test split** strategy, complemented by **cross-validation** to ensure robust evaluation and minimize overfitting. The dataset was randomly shuffled before splitting to maintain class distribution consistency.

### 1.2 Pre-processing
- Converted `.tif` images to `.jpg`
- Resized to `256x256` pixels
- Normalized pixel values
- Applied **mean-std normalization using ImageNet stats** for VGG16 model

### 1.3 Data Augmentation
To enrich training data and reduce overfitting:
- Random horizontal flips
- Random rotations and zooms
- Applied **only on training data**
- Each class was augmented to **500 images**

*![image](https://github.com/user-attachments/assets/c9474e9e-9c66-4947-b2f8-800bfab84b59)*

---

## 2. Neural Networks

### 2.1 Custom CNN Architecture

We developed a **6-layer sequential CNN** with max-pooling, ReLU activations, and dropout layers. The final fully connected layers culminate in a softmax activation for 21-class classification.

![image](https://github.com/user-attachments/assets/fe793361-df3d-4557-92d0-d1e26a9e03f6)


?? **Training and Validation Accuracy**
> *![image](https://github.com/user-attachments/assets/e9da9d37-26ad-4998-8821-4cf9fdc34645)*  


### 2.2 Transfer Learning with VGG16

We used **VGG16 pretrained on ImageNet** for feature extraction and fine-tuning:
- **Feature Extractor Mode**: VGG16 layers frozen; custom classification head trained
- **Fine-Tuning Mode**: Top VGG16 layers unfrozen; jointly trained with classifier

This approach enabled better domain adaptation for aerial imagery.

### 2.3 Training Setup

- **Loss Function**: Categorical Cross-Entropy
- **Optimizer**: Adam with learning rate scheduling
- **Early stopping** applied
- Batch size and epochs tuned empirically
- Performance tracked via training/validation accuracy and loss

?? **Training and Validation Accuracy and Loss for VGG**
> *![image](https://github.com/user-attachments/assets/d0fd8300-3155-48c0-b00f-825711f5d52e)*

?? **Training and Validation Accuracy and Loss for VGG Fine Tuned**
> *![image](https://github.com/user-attachments/assets/88a6f9c1-9da5-49ed-985c-358c3646e35a)*  

---

## 3. Evaluate Models

### 3.1 Test Set Metrics

| Model                    | Accuracy | Precision | Recall | F1-score |
|--------------------------|----------|-----------|--------|----------|
| Custom CNN               | 87%      | 88%       | 87%    | 87%      |
| VGG16 (Feature Extraction) | 80%    | 82%       | 80%    | 79%      |
| VGG16 (Fine-tuned)       | **93%**  | **94%**   | **93%**| **93%**  |

The fine-tuned VGG16 model delivered the best performance.

### 3.2 Confusion Matrix

> *![image](https://github.com/user-attachments/assets/249913d4-050a-48cc-b850-bd3851550997)*  

**Highlights**:
1. Most classes correctly predicted (e.g., 0, 1, 2, 3, 7, 8 show 20/20 correct)
2. Minimal misclassification (e.g., class 12?6, class 17?8)
3. Strong generalization across 21 land-use categories

---

## ?? Conclusion

Fine-tuned VGG16 significantly outperforms the custom CNN in aerial scene classification, achieving 93% accuracy. Augmentation, preprocessing, and selective training strategies contributed to strong model generalization. Future work may include applying attention mechanisms or leveraging more recent transformer-based architectures.

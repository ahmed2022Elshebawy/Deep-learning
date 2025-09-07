# 🐱🐶 Cats vs Dogs Classifier using CNN

This project builds and trains a **Convolutional Neural Network (CNN)** to classify images of cats and dogs using TensorFlow and Keras.

## 📂 Dataset
- Dataset: [Cats and Dogs Filtered](https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip)  
- Split into:
  - **Training set**: 2000 images  
  - **Validation set**: 1000 images  

## ⚙️ Project Workflow
1. **Data Loading & Preparation**
   - Downloaded and extracted dataset
   - Directory structure created for training and validation

2. **Data Preprocessing & Augmentation**
   - Applied rescaling and data augmentation (rotation, zoom, flips, shifts, etc.) on training set
   - Validation set only rescaled

3. **CNN Model Architecture**
   - Conv2D + MaxPooling layers for feature extraction
   - Flatten + Dense layers for classification
   - Output layer with **sigmoid** activation for binary classification

4. **Model Training**
   - Optimizer: Adam
   - Loss: Binary Crossentropy
   - Epochs: 20
   - Training Accuracy: ~72%  
   - Validation Accuracy: ~71%  

5. **Evaluation & Visualization**
   - Plotted accuracy & loss curves
   - Tested model on validation images with predictions

## 📊 Results
- Validation Accuracy: **71.3%**
- Validation Loss: **0.55**

## 🚀 Future Improvements
- Use **transfer learning** with pre-trained models (e.g., VGG16, ResNet50, EfficientNet)  
- Hyperparameter tuning  
- Add dropout & regularization to reduce overfitting  

## 📧 Contact
For any inquiries or collaborations, feel free to reach me at: **ahmed2026shebo@gmail.com**

## 🔗 GitHub Repository
👉 [Project Link](https://github.com/ahmed2022Elshebawy/Deep-learning/commit/084a9d1cd5de83fca5893ac8a20d8c303e3e698f)

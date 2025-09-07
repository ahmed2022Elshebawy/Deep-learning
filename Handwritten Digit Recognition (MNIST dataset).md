#🔢 Handwritten Digit Recognition using CNN

This project builds and trains a **Convolutional Neural Network (CNN)** to classify handwritten digits (0–9) from the MNIST dataset using TensorFlow and Keras.

##📂 Dataset

Dataset: MNIST Handwritten Digits

Split into:

**Training set**: 60,000 images

**Test set**: 10,000 images

##⚙️ Project Workflow

**Data Loading & Preparation**

Loaded MNIST dataset directly from TensorFlow

Reshaped images to (28,28,1) for CNN input

Normalized pixel values (0–1 range)

**Data Visualization**

Displayed random samples of handwritten digits

**CNN Model Architecture**

Conv2D + MaxPooling layers for feature extraction

Dropout layers to reduce overfitting

Flatten + Dense layers for classification

Output layer with softmax activation for 10 classes

**Model Training**

Optimizer: Adam

Loss: Sparse Categorical Crossentropy

Epochs: 20 (with EarlyStopping)

Batch Size: 64

Training Accuracy: ~98%

Validation Accuracy: ~99%

Evaluation & Visualization

Plotted accuracy & loss curves

Generated confusion matrix & classification report

Tested predictions on sample images

##📊 Results

Test Accuracy: 99%

Test Loss: 0.0348

##🚀 Future Improvements

Experiment with deeper CNN architectures

Apply data augmentation for more robust training

Try transfer learning with pre-trained models

Deploy the model as a web app (e.g., Streamlit/Flask)

##📧 Contact

For any inquiries or collaborations, feel free to reach me at: **ahmed2026shebo@gmail.com**

##🔗 GitHub Repository

[Project Link]([https://github.com/ahmed2022Elshebawy/Deep-learning/commit/084a9d1cd5de83fca5893ac8a20d8c303e3e698f](https://github.com/ahmed2022Elshebawy/Deep-learning/blob/main/Handwritten%20Digit%20Recognition%20(MNIST%20dataset).ipynb))

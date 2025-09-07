# Iris Flower Classification using ANN 🌸

This project implements an **Artificial Neural Network (ANN)** using TensorFlow and Keras to classify iris flowers into three species: **Setosa, Versicolor, and Virginica**. The dataset used is the famous [Iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html).

## Steps Covered
1. **Importing Libraries** – Pandas, NumPy, TensorFlow, Scikit-learn.  
2. **Loading Dataset** – Used `sklearn.datasets.load_iris()`.  
3. **Data Preparation** – Converted to DataFrame, checked for missing values, and analyzed class distribution.  
4. **Train-Test Split** – Split data into training (80%) and testing (20%).  
5. **Feature Scaling** – Standardized features using `StandardScaler`.  
6. **Building ANN** – One hidden layer with ReLU activation, and output layer with Softmax activation.  
7. **Training the Model** – Trained for 50 epochs with Adam optimizer.  
8. **Evaluation** – Achieved **97% test accuracy**.  
9. **Prediction** – Made predictions on new flower samples.

## Results
- **Final Test Accuracy**: 97%  
- **Sample Prediction**: Input `[5.1, 3.5, 1.4, 0.2]` → Predicted **Setosa** 🌱  

## Requirements
- Python 3.9+
- TensorFlow
- Scikit-learn
- Pandas
- NumPy

## Notebook
You can explore and run the full notebook here:  
👉 https://github.com/ahmed2022Elshebawy/Deep-learning/blob/main/Iris%20Flower%20Classification%20using%20ANN%20.ipynb
## How to Run
```bash
pip install -r requirements.txt
python iris_ann.py

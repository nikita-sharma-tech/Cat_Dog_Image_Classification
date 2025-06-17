#  Cat vs Dog Image Classification using Deep Learning

Welcome to my **Cat vs Dog Image Classification** project! This is my first project in my journey to becoming an AI Engineer. 
This repository contains a deep learning solution for classifying images of cats and dogs using Convolutional Neural Networks (CNNs) in Python with TensorFlow and Keras.

##  Project Overview

The goal of this project is to build a binary image classifier that accurately distinguishes between images of cats and dogs. This is a common deep learning use case used for understanding CNNs, image preprocessing, and model evaluation.

This project was done as part of my learning journey in computer vision and deep learning.


##  Features

- Image classification using CNN  
- Preprocessing and image resizing  
- Data augmentation for robust model training  
- Training with validation and test accuracy monitoring  
- Evaluation with confusion matrix and performance metrics  
- Built using TensorFlow and Keras  

---

##  Technologies Used

- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Scikit-learn  
- Google Colab  

---

##  Dataset

The dataset used is from Kaggle’s [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats) challenge.  
It consists of 25,000 labeled images (cats and dogs) used for training and testing the model.

---

##  Project Structure

```
├── Cat_Dog_Image_Classification.ipynb  # Jupyter notebook containing full implementation
├── README.md                           # Project description and usage
└── dataset     - > !kaggle competitions download -c dogs-vs-cats
```

##  Step-by-Step Approach

### 1. Import Required Libraries  
All necessary libraries for image handling, model creation, and performance metrics are imported.

### 2. Load and Preprocess Data  
- Images are loaded using Keras’ `ImageDataGenerator`.  
- Image resizing and normalization is performed.  
- Data augmentation is applied to increase dataset diversity.

### 3. Create CNN Model  
- A sequential CNN is built using Conv2D, MaxPooling2D, Flatten, and Dense layers.  
- Activation function: ReLU  
- Output layer uses `sigmoid` activation for binary classification.

### 4. Compile the Model  
- Loss: Binary Crossentropy  
- Optimizer: Adam  
- Metrics: Accuracy

### 5. Train the Model  
- Model is trained using the `fit` method.  
- Accuracy and loss are plotted over epochs for analysis.

### 6. Evaluate the Model  
- Model performance is evaluated using:  
  - Accuracy score  
  - Confusion matrix  
  - Classification report

### 7. Model Testing  
- Test the model on unseen images to verify prediction correctness.

---

##  Results

- Achieved a validation accuracy of **X%** *(fill in your actual accuracy)*  
- Model generalized well on unseen data with minimal overfitting.

---

##  How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/nikita-sharma-tech/cat-dog-classification.git
   cd cat-dog-classification
   ```

2. Open `Cat_Dog_Image_Classification.ipynb` in Jupyter or Google Colab.

3. Ensure you download and extract the dataset into the working directory.

4. Run the notebook step-by-step to train and test the model.

---

##  Learning Outcome

- Gained hands-on experience with image data and CNN architectures.  
- Learned how to handle overfitting with data augmentation.  
- Improved skills in training and evaluating deep learning models.

---

##  Contact

**Nikita Sharma**  
📧 nikitasharmaearthling@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/nikita-sharma-tech)  
🔗 [Kaggle](https://www.kaggle.com/nikitasharmaai)  
🔗 [GitHub](https://github.com/nikita-sharma-tech)



# Potato Leaf Disease Prediction Using Machine Learning

## Project Overview
This project detects potato leaf diseases using Machine Learning and Deep Learning techniques.

The system classifies potato leaf images into three categories:
1. Potato Early Blight
2. Potato Late Blight
3. Potato Healthy

## Objectives
- Detect diseases in potato leaves.
- Preprocess and analyze image datasets.
- Train and compare multiple ML algorithms.
- Develop a simple web application for disease prediction.

## Dataset
Total Images: 301

| Class | Images |
|---|---:|
| Early Blight | 100 |
| Late Blight | 101 |
| Healthy | 100 |
| Total | 301 |

Image Size: 256 × 256 pixels

Image Format: RGB

Dataset Source: Kaggle (add the exact dataset link here).

## Algorithms Used

1. CNN (Convolutional Neural Network)
2. MobileNetV2 (Transfer Learning)
3. SVM (Support Vector Machine with MobileNetV2 feature extraction)

## Model Performance

| Algorithm | Accuracy |
|---|---:|
| CNN | 90.00% |
| MobileNetV2 | 93.33% |
| SVM | 98.33% |

SVM achieved the highest validation accuracy in this experiment.

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Technologies Used
- Python
- TensorFlow
- Keras
- Scikit-learn
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Seaborn

## Project Structure

Crop_Disease_Prediction/
- app/
  - app.py
- dataset/
  - Potato___Early_blight/
  - Potato___Late_blight/
  - Potato___healthy/
- results/
- src/
  - train.py
  - predict.py
  - eda.py
  - visualize.py
  - mobilenet.py
  - evaluate.py
  - evaluate_cnn.py
  - svm_model.py
  - compare_models.py
- README.md

## How to Run

### 1. Install Required Libraries

pip install tensorflow streamlit scikit-learn pandas numpy matplotlib seaborn pillow joblib

### 2. Run the Web Application

From the project root directory:

python -m streamlit run app/app.py

### 3. Upload an Image
Upload a potato leaf image and click Predict Disease.

The application displays the predicted disease.

## Results
The project successfully developed a working prototype for potato leaf disease classification.

The SVM classifier achieved 98.33% validation accuracy on 60 validation images.

Note: These are experimental validation results, not a guarantee of real-world prediction accuracy.

## Future Scope
- Increase dataset size.
- Test on real field images.
- Improve model accuracy and generalization.
- Develop a mobile application.

## Author
Individual B.Tech Project

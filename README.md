# Crop Disease Prediction Using Machine Learning

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

Dataset Source: [PlantVillage Dataset – Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

## Algorithms Used

1. CNN (Convolutional Neural Network)
2. MobileNetV2 (Transfer Learning)
3. SVM (Support Vector Machine with MobileNetV2 feature extraction)

## Model Performance Comparison

All models were evaluated on 60 validation images.

| Algorithm | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| CNN | 90.00% | 92% | 90% | 90% |
| MobileNetV2 | 93.33% | 93% | 93% | 93% |
| SVM | 98.33% | 98% | 98% | 98% |

SVM achieved the highest validation accuracy in this experiment.

Note: These results are based on a small validation dataset and may differ when tested on new real-world images.


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

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/Saheb57/Crop-Disease-Prediction.git
cd Crop-Disease-Prediction
```

### 2. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 3. Run the Web Application

```bash
python -m streamlit run app/app.py
```

### 4. Upload a Leaf Image

- Open the Streamlit application in your browser.
- Upload a potato leaf image.
- Click the Predict Disease button.
- The application displays the predicted disease.

### 5. Dataset for Model Training

Download the PlantVillage Dataset from Kaggle.

Select the required potato disease classes and place the images inside the respective folders in the `dataset/` directory.

## Results
  class_distribution.png
  cnn_confusion_matrix.png
  mobilenetv2_confusion_matrix.png
  model_comparison.csv
  model_comparison.png
  potato_svm_model.pkl
  sample_images.png

## Future Scope
- Increase dataset size.
- Test on real field images.
- Improve model accuracy and generalization.
- Develop a mobile application.

## Author
Individual B.Tech Project

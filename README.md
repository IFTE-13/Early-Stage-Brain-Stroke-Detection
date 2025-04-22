# Predictive Analysis & Early Detection of Brain stroke using Machine Learning Algorithm
This project aims to predict the likelihood of a stroke based on various health and lifestyle factors. The dataset used includes features such as age, average glucose level, BMI, gender, work type, residence type, and smoking status. The project involves exploratory data analysis, data preprocessing, feature engineering, model training, and evaluation.

## Dataset
Find the dataset under the [data.csv](https://github.com/IFTE-13/Early-Stage-Brain-Stroke-Detection/blob/main/data.csv) file.
### Dataset features:
- age: Age of the patient.
- avg_glucose_level: Average glucose level of the patient.
- bmi: Body Mass Index of the patient.
- gender: Gender of the patient.
- work_type: Type of work the patient is engaged in.
- Residence_type: Type of residence (urban or rural).
- smoking_status: Smoking status of the patient.
- stroke: Target variable indicating whether the patient had a stroke (1) or not (0).

## Requirements
### Libraries
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- joblib

## Getting Started
* Clone the repository
```bash
git clone https://github.com/IFTE-13/stroke-prediction.git
```

* Navigate to the project directory
```bash
cd stroke-prediction
```

* Run the script:
```bash
python src/stroke_prediction.py
```

* Install the libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib
```
## About the Model
### Exploratory Data Analysis (EDA)
- Checking the dataset shape.
- Identifying data types.
- Identifying missing values.
- Visualizing distributions and relationships using box plots, histograms, pair plots, and class distribution plots.

### Data Preprocessing
- Handling missing values by filling them with median/mode.
- Encoding categorical variables using LabelEncoder.
- Performing feature engineering (e.g., creating new features like bmi_log, bmi_category, glucose_category, age_group).
- Dropping low-correlation features.

### Principal Component Analysis (PCA)
The script applies PCA to reduce dimensionality and visualize the results.

## Model Training
- K-Nearest Neighbors (KNN)
- Random Forest
- Decision Tree
- Support Vector Machine (SVM)
  
**Hyperparameter tuning is performed using RandomizedSearchCV.**

##Model Evaluation
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- PR AUC

## Results Visualization
The script visualizes the performance of each model using bar plots, confusion matrices, ROC curves, and Precision-Recall curves.

## Final Model Selection
The best model is selected based on the F1 score and saved using joblib.

## Results
The final results are displayed in a table format, and the best model is saved to results/best_stroke_model.pkl.

## Conclusion
This project demonstrates a comprehensive approach to predicting stroke likelihood using machine learning. The steps involved include data preprocessing, feature engineering, model training, and evaluation. The best model is selected based on performance metrics, and the results are visualized for clarity.

## License
> [!CAUTION]
> This project is licensed under the MIT License. Feel free to use and modify the code as per the terms of the license.

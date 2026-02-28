🛒 Online Shoppers Intention Prediction
📌 Project Overview

This project builds a complete Machine Learning pipeline to predict whether an online shopper will generate revenue (make a purchase) based on their browsing session data. The model is trained using the Online Shoppers Intention dataset and applies preprocessing, feature engineering, model selection, and evaluation techniques to achieve accurate predictions.

🎯 Objective

The main objective of this project is to:

Predict customer purchase behavior.

Handle class imbalance effectively.

Compare multiple machine learning models.

Select the best-performing model based on ROC-AUC score.

🛠️ Technologies Used

Python

Pandas

NumPy

Scikit-learn

Imbalanced-learn (SMOTE)

📂 Project Workflow
1️⃣ Data Preprocessing

Converted boolean columns (Weekend, Revenue) into numeric format.

Created a new feature: Returning_Visitor.

Encoded categorical variables.

Performed correlation analysis.

2️⃣ Train-Test Split

Split dataset into 70% training and 30% testing data.

3️⃣ Machine Learning Pipeline

The pipeline includes:

Missing value imputation

MinMax Scaling

One-Hot Encoding

SMOTE (for class imbalance handling)

Feature selection using SelectKBest (Chi-Square test)

Model training

4️⃣ Models Compared

Random Forest Classifier

Decision Tree Classifier

K-Nearest Neighbors

Ridge Classifier

Bernoulli Naive Bayes

Support Vector Classifier (SVC)

Models are evaluated using 10-fold cross-validation with ROC-AUC score.

📊 Evaluation Metrics

ROC-AUC Score

Accuracy

F1 Score

Classification Report

🚀 How to Run the Project

Install required libraries:

pip install pandas numpy scikit-learn imbalanced-learn

Place online_shoppers_intention.csv in the project directory.

Run the script:

python final_code_updated.py
📈 Results

The best-performing model is selected based on the highest ROC-AUC score and evaluated on the test dataset.

👨‍💻 Author

VAJJALA SOMESWARA RAO

Developed as a Machine Learning project to demonstrate end-to-end pipeline implementation, model comparison, and evaluation techniques.

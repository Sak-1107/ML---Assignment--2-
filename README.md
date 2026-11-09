# ML---Assignment--2-
ML Assignment 2 - Bits wilp Mtech AIML 

\# Customer Churn Prediction Using Machine Learning



\## 1. Problem Statement



Customer churn is a major challenge in the telecommunications industry. Retaining existing customers is important for business growth and profitability.



The objective of this project is to develop machine learning classification models to predict whether a customer is likely to leave their current telecommunications service provider.



Five classification algorithms are implemented and evaluated using multiple performance metrics. An interactive Streamlit application is developed to allow users to upload test data, select a trained model and view its performance.



\## 2. Dataset Description



\*\*Dataset Name:\*\* Customer Churn Prediction 2020



\*\*Source:\*\* \[Kaggle – Customer Churn Prediction 2020](https://www.kaggle.com/competitions/customer-churn-prediction-2020/data)



\*\*Dataset Details:\*\*

\- Total Instances: 4,250

\- Total Columns: 20

\- Input Features: 19

\- Target Variable: `churn`

\- Numerical Features: 15

\- Categorical Features: 4

\- Missing Values: 0

\- Duplicate Records: 0



The dataset contains customer information such as account length, area code, international calling plan, voice mail plan, day/evening/night call usage, call charges and customer service calls.



The target variable `churn` indicates whether a customer has left the service provider:

\- `No` – Customer retained

\- `Yes` – Customer churned



\*\*Class Distribution:\*\*

\- No Churn: 3,652 (85.93%)

\- Churn: 598 (14.07%)



The dataset is imbalanced, with a higher proportion of non-churning customers.



\### Data Preprocessing



The following preprocessing techniques were applied:

\- Target variable encoding (`no` = 0, `yes` = 1).

\- Numerical features standardized using StandardScaler.

\- Categorical features encoded using OneHotEncoder.

\- Stratified train-test split (70:30).

\- Random state: 42.



The training dataset contains 2,975 records, while the test dataset contains 1,275 records.



\## 3. GitHub Repository Link



\*\*GitHub Repository:\*\* \[Add your GitHub repository URL here]



The repository contains:

\- `app.py` – Streamlit application.

\- `train\_models.py` – Model training and evaluation script.

\- `requirements.txt` – Required Python dependencies.

\- `test\_data.csv` – Test dataset for application evaluation.

\- `model/` – Saved trained models.

\- `results/model\_comparison.csv` – Model evaluation results.



\## 4. Models Used and Evaluation



The following five classification algorithms were implemented using the same dataset:



1\. Logistic Regression

2\. Decision Tree Classifier

3\. K-Nearest Neighbors (KNN)

4\. Gaussian Naive Bayes

5\. Support Vector Machine (SVM)



\### Evaluation Metrics



The models were evaluated using the following metrics:



\- \*\*Accuracy:\*\* Measures the overall proportion of correct predictions.

\- \*\*AUC Score:\*\* Measures the model's ability to distinguish between positive and negative classes.

\- \*\*Precision:\*\* Measures the proportion of predicted churn cases that were actually churn cases.

\- \*\*Recall:\*\* Measures the proportion of actual churn cases correctly identified.

\- \*\*F1 Score:\*\* Harmonic mean of precision and recall.

\- \*\*MCC:\*\* Matthews Correlation Coefficient, which considers all four confusion matrix outcomes.



\### Model Comparison Table



| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |

|---|---:|---:|---:|---:|---:|---:|

| Logistic Regression | 0.8612 | 0.8081 | 0.5104 | 0.2737 | 0.3564 | 0.3039 |

| Decision Tree | 0.9176 | 0.8352 | 0.7011 | 0.7207 | 0.7107 | 0.6628 |

| KNN | 0.8933 | 0.8141 | 0.8413 | 0.2961 | 0.4380 | 0.4600 |

| Naive Bayes | 0.5922 | 0.6604 | 0.2065 | 0.6704 | 0.3158 | 0.1742 |

| Support Vector Machine | 0.9161 | 0.8986 | 0.8830 | 0.4637 | 0.6081 | 0.6031 |



\### Observations on Model Performance



\*\*1. Logistic Regression\*\*



Logistic Regression achieved an accuracy of 86.12% and an AUC score of 0.8081. However, its recall of 27.37% indicates that it failed to identify a considerable proportion of actual churn customers.



\*\*2. Decision Tree Classifier\*\*



Decision Tree achieved an accuracy of 91.76% and the highest F1 score of 0.7107 among all five models. It also achieved the highest recall and MCC, indicating a comparatively balanced performance in identifying churn and non-churn customers.



\*\*3. K-Nearest Neighbors (KNN)\*\*



KNN achieved a precision of 84.13%, indicating good reliability when predicting churn. However, its recall was relatively low at 29.61%, meaning that it missed many actual churn customers.



\*\*4. Gaussian Naive Bayes\*\*



Naive Bayes achieved a recall of 67.04%, but its precision was only 20.65%. This indicates that although it identified many actual churn customers, it also generated a high number of false positive predictions.



\*\*5. Support Vector Machine (SVM)\*\*



SVM achieved the highest AUC score of 0.8986 and the highest precision of 88.30%. However, its recall of 46.37% indicates that it missed a significant proportion of actual churn customers.



\### Overall Winner



\*\*Decision Tree Classifier\*\*



Based on the evaluation results, Decision Tree is the overall winner for this dataset.



It achieved:

\- Highest Accuracy: 91.76% (slightly higher than SVM).

\- Highest Recall: 72.07%.

\- Highest F1 Score: 0.7107.

\- Highest MCC: 0.6628.



Although SVM achieved the highest AUC and precision, Decision Tree provided a better balance between precision and recall, making it the strongest overall performer for this classification problem.



\## 5. Streamlit Application



An interactive web application was developed using Streamlit to demonstrate the performance of all five trained classification models.



\*\*Live Streamlit Application:\*\* \[Add your deployed Streamlit URL here]



\### Application Features



1\. \*\*Dataset Upload:\*\* Users can upload a CSV file containing the test records, including all required input features and the target column.

2\. \*\*Model Selection:\*\* Users can select any one of the five trained classification models through a dropdown menu.

3\. \*\*Evaluation Metrics:\*\* The application calculates and displays Accuracy, AUC, Precision, Recall, F1 Score and MCC.

4\. \*\*Confusion Matrix:\*\* Displays the confusion matrix and classification report for the selected model.

5\. \*\*Model Comparison:\*\* Enables users to evaluate and compare the performance of different classification algorithms.



The application loads the previously trained models and evaluates them on the uploaded test dataset without retraining.



\## 6. Technologies and Libraries Used



\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- Streamlit

\- Matplotlib

\- Seaborn

\- Joblib



\## 7. How to Run the Project Locally



\*\*Step 1:\*\* Clone the repository.



```bash

git clone YOUR\_GITHUB\_REPOSITORY\_URL

```



\*\*Step 2:\*\* Navigate to the project directory.



```bash

cd BITS-ML-Assignment-2

```



\*\*Step 3:\*\* Install the dependencies.



```bash

pip install -r requirements.txt

```



\*\*Step 4:\*\* Launch the Streamlit application.



```bash

streamlit run app.py

```



The application will open in your default web browser.



\---

\*\*Name:\*\* Sakshi patil

\*\*ID:\*\* 2026AJ05633



\*\*Academic Programme:\*\* M.Tech Artificial Intelligence and Machine Learning



\*\*Institution:\*\* BITS Pilani – Work Integrated Learning Programmes Division



\*\*Course:\*\* Machine Learning



\*\*Assignment:\*\* Assignment 2 – Classification Models and Streamlit Deployment



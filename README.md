
# Team Number –AG2 -An Explainable Deep Learning Framework for Predicting Traffic Accident Risk Using CNN–BiLSTM–Attention and SHAP
## Team Info
- 22471A0526 — **Kallam Thirupatamma**  ( [LinkedIn](https://www.linkedin.com/in/thirupatamma-kallam-095271301/))
_Work Done: Designed and implemented the complete CNN–BiLSTM–Attention model for traffic accident severity prediction. Performed data preprocessing, feature engineering, 10-fold cross-validation, DeepSHAP explainability analysis, performance evaluation (Accuracy, Precision, Recall, F1-score), and research paper documentation.

- 22471A0504 — **Biraka Velangini Rani** ( [LinkedIn](https://www.linkedin.com/in/velangini-rani) )
_Work Done: Collected and cleaned the UK road traffic accident dataset. Conducted exploratory data analysis (EDA), handled missing values, performed feature scaling and encoding, visualized severity distributions, and assisted in comparative model evaluation with Logistic Regression.

- 22471A0569 — **Yenumula Mythri  **( [LinkedIn](https://www.linkedin.com/in/mythri-yenumala-8a396b326/) )
_Work Done: Conducted literature survey and analysis of IEEE base paper. Assisted in documentation, result interpretation, confusion matrix analysis, SHAP feature importance visualization, deployment support using Flask, and presentation preparation.
---

## Abstract
Accurate estimation of accident risk and the identification of critical contributing factors are critical for improving traffic safety and road fatality reduction. This paper introduces a new deep learning architecture that combines Convolutional Neural Networks (CNN), Bidirectional Long Short-Term Memory (BiLSTM), and Attention to capture both spatial and temporal trends in traffic accident data. To provide interpretability and credibility for decision-making, the model is augmented further with SHAP (Shapley Additive explanations) to identify the most significant features responsible for accident severity. The model is tested on two real datasets: the US Accident Dataset (March 2023) and the UK STATS19 dataset, following rigorous preprocessing, feature normalization, and class balancing using SMOTE Tomek and RandomOverSampler methods. The suggested CNN– BiLSTM–Attention model presents high predictive accuracy, MAE of 0.137 (US) and 0.179 (UK), far exceeding conventional models such as Logistic Regression. Through SHAP analysis, the framework reveals transparent insights regarding the key factors determining accident severity, and the framework is both precise and interpretable. This paper presents an efficient andscalable solution for real time traffic risk prediction and traffic authority decision support.

---

## Paper Reference (Inspiration)
👉 **[Paper Title : Road Traffic Accident Risk Prediction and Key Factor Identification Framework Based on Explainable Deep Learning
  – Author Names:Yulong Pei; Yuhang Wen; Sheng Pan
 ](https://ieeexplore.ieee.org/document/10658644/;jsessionid=665DCAA5DD019C3DD45FE722647BFC13)**
Original conference/IEEE paper used as inspiration for the model.

---

## Our Improvement Over Existing Paper
Implemented full end-to-end pipeline (data preprocessing → modeling → explainability → deployment).

✔ Applied 10-fold Cross-Validation for robust evaluation instead of single train-test split.

✔ Integrated Flask-based web deployment for real-time prediction support.

✔ Added comparative analysis with Logistic Regression for performance benchmarking.

✔ Improved feature engineering and class balancing using SMOTETomek and RandomOverSampler.

✔ Optimized implementation for faster execution suitable for Google Colab and real-time usage.
---

## About the Project
Give a simple explanation of:
- What your project does
  
🔹 Why it is useful
- General project workflow (input → processing → model → output)
This project predicts the severity of road traffic accidents (Slight, Serious, Fatal) using a deep learning model. It also explains which features contribute most to the prediction using SHAP.

🔹 Why It Is Useful

Helps traffic authorities identify high-risk scenarios

Supports decision-making for road safety policies

Enables proactive accident prevention strategies

Provides transparent AI decisions for better trust

🔹 General Workflow
Accident Data (US / UK)
→ Data Cleaning & Preprocessing
→ Feature Engineering & Normalization
→ Class Balancing (SMOTE Tomek / RandomOverSampler)
→ CNN–BiLSTM–Attention Model
→ Severity Prediction
→ SHAP Explainability (Key Risk Factors)
→ Deployment via Flask
---

## Dataset Used
👉 **[UK Road Safety Accidents & Vehicles Dataset](https://www.kaggle.com/datasets/tsiaras/uk-road-safety-accidents-and-vehicles?utm_source=chatgpt.com)**
👉 **[US Accidents Dataset (2016–2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)**

**Dataset Details:**
• Real-world large-scale accident datasets
• Structured accident, vehicle, and environmental features
• Includes weather, road type, speed limit, visibility, vehicle type, etc.
• Multi-class severity labels

---

## Dependencies Used
Python, PyTorch, TensorFlow, NumPy, Pandas, Scikit-learn, Matplotlib, SHAP, Imbalanced-learn, Joblib, Flask

---

## EDA & Preprocessing
🧹 Removed missing and inconsistent records
🏷 Encoded categorical features using label encoding
📏 Normalized numerical features using standard scaling
⚖ Balanced class distribution using SMOTETomek
🕒 Extracted temporal features such as hour, day, and month

---

## Model Training Info
🧠 CNN extracts spatial feature representations
🔁 BiLSTM learns bidirectional temporal patterns
🎯 Attention mechanism highlights important time steps
📉 Cross-entropy loss used for optimization
⚡ Adam optimizer improves convergence
---

## Model Testing / Evaluation
The model performance is evaluated using:
Accuracy
Precision
Recall
F1-score
Confusion Matrix
We used 10-Fold Cross-Validation to ensure reliable results.

---

## Results
✅ US Dataset

Accuracy: ~87%

MAE: 0.137

High recall for severe accident cases

✅ UK Dataset

Accuracy: ~86%

MAE: 0.179

Improved F1-score compared to Logistic Regression

🔎 SHAP Insights

Weather conditions

Lighting conditions

Speed

Road surface

Time of accident

These were identified as major contributing factors influencing accident severity

---

## Limitations & Future Work
💻 Performance depends on data quality
📉 Rare fatal cases remain challenging
🌐 Future Enhancements:

Real-time accident prediction

Integration with live traffic sensors

Deployment on cloud platforms

Advanced spatio-temporal modeling

---

## Deployment Info
🖥 Implemented using Flask
⚡ Model packaged into a single deployable PKL
📊 Supports manual and bulk CSV prediction
🔍 Stores prediction history in database
---
## Project By

Thirupatamma Kallam
An Explainable Deep Learning Framework for Traffic Accident Risk Prediction

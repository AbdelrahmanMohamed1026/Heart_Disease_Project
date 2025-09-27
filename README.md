# 🫀 Heart Disease Prediction – Machine Learning Pipeline 
Comprehensive Machine Learning Full Pipeline on Heart Disease UCI Dataset

## 📌 Project Overview  
This project implements a **comprehensive machine learning pipeline** on the **Heart Disease UCI dataset**.  
The goal is to analyze, predict, and visualize the risk of heart disease through end-to-end steps: preprocessing, dimensionality reduction, feature selection, model training, evaluation, and deployment with a Streamlit web UI.  

------

## 🚀 Workflow  

### **2.1 Data Preprocessing**  
- Handled missing values  
- Encoded categorical features using **One-Hot Encoding**  
- Binarized target: `num > 0 → 1` (Disease vs No Disease)  
- Standardized features using **MinMaxScaling**  

### **2.2 Dimensionality Reduction (PCA)**  
- Applied **Principal Component Analysis (PCA)**  
- Analyzed explained variance ratio  
- Visualized cumulative variance 

### **2.3 Feature Selection**  
- Ranked features by importance using **Random Forest**, **RFECV**, **Chi-Square test**
- Selected final subset of features for modeling  

### **2.4 Supervised Learning**  
- Trained multiple classifiers:  
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - Support Vector Machine (SVM)  
- Evaluated models using:  
  - Accuracy, Precision, Recall, F1-score  
  - AUC & ROC curve  

### **2.5 Unsupervised Learning**  
- Applied **KMeans clustering**  
- Used the **Elbow method** to find optimal clusters  

### **2.6 Hyperparameter Tuning**  
- Used **GridSearchCV** & **RandomizedSearchCV**  
- Best model: **SVM** with optimized `C`, `gamma`, `kernel`, and `degree`  

### **2.7 Model Export**  
- Built an end-to-end pipeline:  
  *(Preprocessing → Feature Selection →  Model)*  
- Exported with **joblib** as:  final_model_pipeline.pkl

### **2.8 Deployment (Streamlit UI)**  
- Built a **simple form-based web interface**  
- Users can input patient data  
- Model outputs:  
- Prediction (Disease / No Disease)  
- Probability score

------

## 📂 Project Structure  
``` bash
Heart_Disease_Project/
│── data/
│ └── heart_disease.csv
│── notebooks/
│ ├── 01_data_preprocessing.ipynb
│ ├── 02_pca_analysis.ipynb
│ ├── 03_feature_selection.ipynb
│ ├── 04_supervised_learning.ipynb
│ ├── 05_unsupervised_learning.ipynb
│ ├── 06_hyperparameter_tuning.ipynb
│── models/
│ └── final_model_pipeline.pkl
│── ui/
│ └── app.py # Streamlit interface
│── results/
│ └── evaluation_metrics.txt
│── README.md
│── requirements.txt
│── .gitignore
```

------

## Quick Start / Installation  

1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```bash
   python -m streamlit run ui/app.py
   ```
   
 ------

 ## 📊 Final Deliverables  
✔️ Cleaned & processed dataset  
✔️ PCA analysis & variance visualization  
✔️ Feature importance ranking  
✔️ Trained classification models  
✔️ Performance evaluation (Accuracy, Precision, Recall, F1-score, AUC)  
✔️ Hyperparameter-optimized **SVM model**  
✔️ Exported pipeline (`.pkl`)  
✔️ Streamlit app for real-time prediction  

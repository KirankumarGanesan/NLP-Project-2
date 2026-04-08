# NLP Supervised Learning: Insurer Review Analytics 🛡️

Intelligent analysis of customer reviews in the insurance sector to automate star-rating predictions and categorize feedback.

## 👥 Project Team
* **Kiran Kumar GANESAN**
* **Rithiga VENGADESSANE**

## 📊 Project Overview
This project processes a dataset of over 34,000 insurance reviews to build a supervised NLP pipeline. The core objective is predicting the user's Star Rating (1-5 stars) based on the textual content of their review. 

**Key Phases:**
1. **Data Exploration & Cleaning:** Regex-based standardization, bi-gram analysis, and fast-text cleaning.
2. **Supervised Learning:** TF-IDF Vectorization paired with a Random Forest Classifier.
3. **Dashboarding:** An interactive Streamlit application allowing real-time predictions and company metric visualization.

---

## ⚠️ Data Quality Note: Missing Dates
During the Data Exploration phase, we identified that several entries in the raw dataset were missing chronological data (specifically in the `date_publication` or `date_exp` columns). 

**Handling Strategy:**
Because the core objective of this supervised learning model is NLP-based (mapping text patterns to Star Ratings), chronological features were not required for the Random Forest classifier. Therefore, reviews with missing dates were consciously retained. This decision maximized the training volume for our TF-IDF vectorizer, ensuring the model had the largest possible vocabulary to learn from.

---

## 📁 Local vs. Repository Architecture (Excluded Files)
To adhere to GitHub's file size limits (100 MB) and professional Data Engineering best practices, several large and locally-generated files have been explicitly excluded from this remote repository using `.gitignore`.

**Files kept exclusively on local machines:**
* **`src/star_model.pkl` (125.67 MB):** The serialized Random Forest classifier.
* **`src/tfidf_vectorizer.pkl`:** The trained NLP text vectorizer.
* **`processed_data.csv`:** The merged and cleaned dataset generated from the raw translation files.
* **`Demo video .mp4`:** The local recording of our Streamlit dashboard presentation.

---

## 🚀 How to Reproduce & Run the Application
Because this repository contains the complete pipeline logic, the `.pkl` models and `.csv` dataset will be automatically generated on your local machine by running the scripts.

### 1. Install Dependencies
Ensure you have the required libraries installed:
```bash
pip install streamlit pandas openpyxl scikit-learn joblib

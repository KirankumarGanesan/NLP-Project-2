import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# 1. DYNAMIC PATH SELECTION
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
csv_path = os.path.join(project_root, "processed_data.csv")

# 2. LOAD DATA
print("Loading processed data...")
df = pd.read_csv(csv_path)
df = df.dropna(subset=['cleaned_text', 'note'])

# 3. TF-IDF VECTORIZATION
print("Vectorizing text with TF-IDF...")
tfidf = TfidfVectorizer(max_features=1000, token_pattern=r"(?u)\b\w+\b")
X = tfidf.fit_transform(df['cleaned_text'].astype(str))
y = df['note'].astype(int)

# 4. SUPERVISED LEARNING (Random Forest)
print("Training the Supervised Model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 5. SAVE MODELS IN /src/
joblib.dump(model, os.path.join(script_dir, 'star_model.pkl'))
joblib.dump(tfidf, os.path.join(script_dir, 'tfidf_vectorizer.pkl'))
print("SUCCESS: Models trained and saved in /src/ folder!")
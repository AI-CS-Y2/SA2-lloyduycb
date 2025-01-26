import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv('/kaggle/input/full-apple-tv-dataset/data.csv')

# Display overview
print("Dataset Overview: \n")
print(data.head(), "\n\n")
print("Dataset Info: \n")
print(data.info(), "\n\n")
print("Summary Statistics: \n")
print(data.describe(), "\n\n")

# Fill missing values
data['genres'] = data['genres'].fillna('')  # Empty string for genres
data['releaseYear'] = data['releaseYear'].fillna(data['releaseYear'].median())
data['imdbAverageRating'] = data['imdbAverageRating'].fillna(data['imdbAverageRating'].median())
data['imdbNumVotes'] = data['imdbNumVotes'].fillna(data['imdbNumVotes'].median())
data['availableCountries'] = data['availableCountries'].fillna('')

# Feature engineering
data['is_drama'] = data['genres'].apply(lambda x: 1 if 'Drama' in x else 0)
data['country_count'] = data['availableCountries'].apply(lambda x: len(x.split(',')))

# Normalize numerical features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
numerical_features = ['releaseYear', 'imdbAverageRating', 'imdbNumVotes', 'country_count']
data[numerical_features] = scaler.fit_transform(data[numerical_features])


import matplotlib.pyplot as plt
import seaborn as sns

# Distribution of IMDb ratings
plt.figure(figsize=(8, 6))
sns.histplot(data['imdbAverageRating'], bins=30, kde=True)
plt.title('Distribution of IMDb Ratings')
plt.xlabel('IMDb Rating')
plt.ylabel('Frequency')
plt.show()

# Top 10 genres
from collections import Counter
all_genres = [genre for sublist in data['genres'].apply(lambda x: x.split(', ')) for genre in sublist]
genre_counts = Counter(all_genres)
top_genres = genre_counts.most_common(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=[g[0] for g in top_genres], y=[g[1] for g in top_genres])
plt.title('Top 10 Genres')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


from sklearn.model_selection import train_test_split

# Features and target
features = ['releaseYear', 'imdbAverageRating', 'imdbNumVotes', 'country_count']
target = 'is_drama'
X = data[features]
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Decision Tree
dt_model = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

# Logistic Regression
lr_model = LogisticRegression(solver='liblinear', random_state=42)
lr_model.fit(X_train, y_train)


from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    log_loss, confusion_matrix
)

# Function to evaluate models
def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, proba)
    logloss = log_loss(y_test, proba)
    conf_matrix = confusion_matrix(y_test, predictions)

    print(f"--- {name} Evaluation --- \n")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print(f"ROC-AUC: {roc_auc:.2f}")
    print(f"Log Loss: {logloss:.2f}")
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "Log Loss": logloss,
    }

# Evaluate Decision Tree
dt_results = evaluate_model("Decision Tree", dt_model, X_test, y_test)

# Evaluate Logistic Regression
lr_results = evaluate_model("Logistic Regression", lr_model, X_test, y_test)



import matplotlib.pyplot as plt

# Compare evaluation metrics
results = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Log Loss"],
    "Decision Tree": [dt_results["Accuracy"], dt_results["Precision"], dt_results["Recall"],
                      dt_results["F1-Score"], dt_results["ROC-AUC"], dt_results["Log Loss"]],
    "Logistic Regression": [lr_results["Accuracy"], lr_results["Precision"], lr_results["Recall"],
                             lr_results["F1-Score"], lr_results["ROC-AUC"], lr_results["Log Loss"]]
})

print("Evaluation Metrics Comparison: \n")
print(results, "\n")

# Bar plot for metrics comparison
results.set_index("Metric").plot(kind="bar", figsize=(10, 6))
plt.title("Comparison of Evaluation Metrics")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.show()

# Visualize confusion matrices
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_test, dt_model.predict(X_test), display_labels=["Non-Drama", "Drama"])
plt.title("Decision Tree Confusion Matrix")
plt.show()

ConfusionMatrixDisplay.from_predictions(y_test, lr_model.predict(X_test), display_labels=["Non-Drama", "Drama"])
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# Ensure probabilities are defined
dt_proba = dt_model.predict_proba(X_test)[:, 1]
lr_proba = lr_model.predict_proba(X_test)[:, 1]

# Visualize Precision-Recall Curve for both models
from sklearn.metrics import precision_recall_curve

# Get precision and recall values for Decision Tree
dt_precision, dt_recall, _ = precision_recall_curve(y_test, dt_proba)

# Get precision and recall values for Logistic Regression
lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_proba)

# Plot the Precision-Recall Curve
plt.figure(figsize=(10, 6))
plt.plot(dt_recall, dt_precision, label="Decision Tree", linewidth=2)
plt.plot(lr_recall, lr_precision, label="Logistic Regression", linewidth=2)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)
plt.show()



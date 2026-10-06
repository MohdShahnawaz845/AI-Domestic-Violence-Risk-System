import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# 1. Dataset load karna
data = pd.read_csv("dataset/data.csv")

print("Dataset successfully loaded!")
print(data.head())


# 2. Features aur target alag karna
X = data.drop("risk_level", axis=1)
y = data["risk_level"]


# 3. Training aur testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 4. Random Forest model banana
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# 5. Model train karna
model.fit(X_train, y_train)

print("Model training completed!")


# 6. Prediction
y_pred = model.predict(X_test)


# 7. Accuracy
accuracy = accuracy_score(y_test, y_pred)
with open("model/accuracy.txt", "w") as file:
    file.write(str(accuracy))

print("Model Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# 8. Model save karna
joblib.dump(model, "model/domestic_violence_model.pkl")

print("Model saved successfully!")
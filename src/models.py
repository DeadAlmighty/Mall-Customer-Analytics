# Machine Learning models module
"""
This module contains machine learning model implementations.
"""

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def train_decision_tree(df):
    """
    Train Decision Tree classifier
    """
    X = df[['Age', 'Annual Income (k$)']]
    y = df['Cluster']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    return y_test, y_pred


def evaluate_model(y_test, y_pred):
    """
    Evaluate model performance
    """
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
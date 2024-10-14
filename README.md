# anomadata

  1. Import Libraries:
Start by importing necessary Python libraries for data manipulation, visualization, and model building.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import joblib

  2. Load and Explore the Data:
Load your dataset and perform an initial exploration to understand the shape and distribution of the data.
df = pd.read_csv('C:/Users/Admin/OneDrive/Desktop/Data Science/AnomaData - Capstone.csv')

  # Check data structure
print(df.head())
print(df.info())

  3. Data Cleaning:
Handle missing values by imputing them. In this project, missing numerical values are imputed using the mean strategy.
numeric_columns = df.select_dtypes(include=[np.number]).columns
imputer = SimpleImputer(strategy='mean')
df_cleaned[numeric_columns] = imputer.fit_transform(df[numeric_columns])

  4. Handle Imbalanced Data:
Use SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset by oversampling the minority class.
X = df_cleaned.drop('target_column', axis=1)
y = df_cleaned['target_column']

  # Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

  # Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

  5. Model Selection and Training:
We use a Random Forest classifier to train the model. The best parameters are found using GridSearchCV.
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'min_samples_split': [2, 5, 10],
    'max_depth': [None, 10, 20, 30]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train_smote, y_train_smote)

# Print best parameters
print(f"Best parameters: {grid_search.best_params_}")

  6. Model Evaluation:
Evaluate the model using the test set and generate performance metrics such as a confusion matrix and classification report.
y_pred = grid_search.best_estimator_.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

   7. Model Deployment:
Once the model is trained and evaluated, save the best performing model using joblib for future use.
best_rf_model = grid_search.best_estimator_
joblib.dump(best_rf_model, 'anomaly_detection_model.pkl')

  # Confirm model saving
print("Model saved as 'anomaly_detection_model.pkl'.")

Deployment:
To deploy the model for inference, load the saved model using joblib:
import joblib

  # Load the saved model
model = joblib.load('anomaly_detection_model.pkl')

  # Use the model for prediction
predictions = model.predict(new_data)

Conclusion:
This project demonstrates the end-to-end process of building an anomaly detection model. The steps cover data cleaning, handling imbalanced datasets using SMOTE, model training and evaluation using RandomForest, and deployment of the final model.



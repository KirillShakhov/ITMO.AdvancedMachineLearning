
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split

# Load the training data
train_data = pd.read_csv('data/titanic_train.csv')

# Handling missing values for 'age'
age_imputer = SimpleImputer(strategy='mean')
train_data['age'] = age_imputer.fit_transform(train_data['age'].values.reshape(-1, 1))

# One-hot encoding for 'sex' and 'embarked'
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_features = encoder.fit_transform(train_data[['sex', 'embarked']])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['sex', 'embarked']))

# Concatenating encoded features and dropping original 'sex' and 'embarked' columns
train_data_encoded = pd.concat([train_data, encoded_df], axis=1).drop(['sex', 'embarked', 'name', 'ticket', 'cabin', 'home.dest'], axis=1)

# Separating features and target variable
X = train_data_encoded.drop('survived', axis=1)
y = train_data_encoded['survived']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initializing the Random Forest Classifier and fitting it to the training data
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Predicting the outcomes for the test set
rf_predictions = rf.predict(X_test)

# Evaluating the model
rf_f1_score = f1_score(y_test, rf_predictions)
rf_classification_report = classification_report(y_test, rf_predictions)

# Displaying the evaluation results
print(f"Random Forest Classifier - F1 Score: {rf_f1_score}")
print(f"Classification Report: \n{rf_classification_report}")

# Predicting with the model (example)
new_data_predictions = rf.predict(X_test.head())
print(f"Example Predictions: {new_data_predictions.tolist()}")

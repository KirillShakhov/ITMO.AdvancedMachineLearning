import pandas as pd
from sklearn.svm import SVC

# Load the training data
train_data_path = 'persons_pics_train.csv'  # Update the path as needed
train_data = pd.read_csv(train_data_path)
x_train = train_data.drop('label', axis=1)
y_train = train_data['label']

# Load the test data
test_data_path = 'persons_pics_reserved.csv'  # Update the path as needed
x_test = pd.read_csv(test_data_path)

# Train the SVC model
svc = SVC(kernel='linear', gamma=1e-3, C=1, class_weight=None, probability=True)
svc.fit(x_train, y_train)

# Predict labels for the test data
predicted_labels = svc.predict(x_test)

# Print the predicted labels
print(predicted_labels.tolist())

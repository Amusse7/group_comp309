# create_test_model.py
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np

# Create a simple dummy model
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y = np.array([0, 1, 0])

# Train a simple model
model = LogisticRegression()
model.fit(X, y)

# Save the model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Test model has been created and saved as 'model.pkl'")

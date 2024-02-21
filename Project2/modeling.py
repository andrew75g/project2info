from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training with hyperparameter tuning
log_reg_params = {'C': [0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(LogisticRegression(), log_reg_params, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

best_log_reg = grid_search.best_estimator_
print("Best Logistic Regression Parameters:", grid_search.best_params_)

# Predicting on the test set
y_pred = best_log_reg.predict(X_test_scaled)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Logistic Regression Test Accuracy:", accuracy)

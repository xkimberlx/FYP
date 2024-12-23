import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import re
import joblib
import numpy as np

# Function to extract 'Near KTM/LRT'
def extract_near_ktm_lrt(text):
    pattern = re.compile(r'\bNear KTM/LRT\b')
    try:
        match = pattern.search(str(text))  # Ensure text is a string
        if match:
            return 'yes'
        return 'no'
    except TypeError:
        return 'no'

# Load the dataset
df = pd.read_csv('preprocessed3_rent_pricing_kuala_lumpur.csv')

def map_require_facilities(facilities):
    if isinstance(facilities, str) and facilities.lower() != 'None':
        return 1
    return 0

df['require_facilities'] = df['facilities'].apply(map_require_facilities)

# Convert 'rooms' column to numeric (int) to avoid comparison issues
df['rooms'] = pd.to_numeric(df['rooms'], errors='coerce')

# Filter out rows where the 'rooms' column has values greater than 10
df = df[df['rooms'] <= 10]

# Extract 'Near KTM/LRT' from the 'additional_facilities' column and create a new column
df['near_ktm_lrt'] = df['additional_facilities'].apply(extract_near_ktm_lrt)

# Convert the 'near_ktm_lrt' column to numeric: 'yes' -> 1, 'no' -> 0
df['near_ktm_lrt'] = df['near_ktm_lrt'].map({'yes': 1, 'no': 0})

# Scale numerical features
scaler = StandardScaler()
df[['size_sqft', 'rooms']] = scaler.fit_transform(df[['size_sqft', 'rooms']])

# Save the scaler for later use in deployment
joblib.dump(scaler, 'scaler.pkl')

# Add region column to the dataset (assuming it's part of your data or will be added manually)
df['region'] = df['region'].map({'Kuala Lumpur': 1, 'Selangor': 0})  # Mapping region to 1 (KL) and 0 (Selangor)

# Focus on relevant columns, including the newly created 'near_ktm_lrt'
X = df[['size_sqft', 'furnished', 'rooms', 'near_ktm_lrt', 'property_type', 'parking', 'region', 'require_facilities']]

# Convert parking column: 0 -> 0 (No Parking), 1-10 -> 1 (Yes)
df['parking'] = df['parking'].apply(lambda x: 0 if x == 0 else 1)

# One-hot encode the 'furnished' and 'property_type' columns
X = pd.get_dummies(X, columns=['furnished', 'property_type'], drop_first=False)

# Target variable
y = df['monthly_rent']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Regressor (before random search)
rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(X_train, y_train)

# Predict on the test set
y_pred_test = rf_reg.predict(X_test)

# Calculate evaluation metrics (before random search)
mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

# Print evaluation metrics before RandomizedSearchCV
print(f"\nTest Metrics (Before Randomized Search):")
print(f"Mean Squared Error (MSE): {mse_test:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_test:.2f}")
print(f"Mean Absolute Error (MAE): {mae_test:.2f}")
print(f"R² Score: {r2_test:.4f}")

# Hyperparameter grid for RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 200, 500, 1000],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# RandomizedSearchCV to tune the hyperparameters
random_search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=50,  # Number of random samples to evaluate
    cv=5,  # 3-fold cross-validation
    scoring='neg_mean_squared_error',
    verbose=1,
    random_state=42,
    n_jobs=-1  # Use all available CPU cores
)

# Fit the RandomizedSearchCV
random_search.fit(X_train, y_train)

# Get the best estimator
best_rf_reg = random_search.best_estimator_

# Predict on the test set (after random search)
y_pred_test_random = best_rf_reg.predict(X_test)

# Calculate evaluation metrics (after random search)
mse_test_random = mean_squared_error(y_test, y_pred_test_random)
rmse_test_random = np.sqrt(mse_test_random)
mae_test_random = mean_absolute_error(y_test, y_pred_test_random)
r2_test_random = r2_score(y_test, y_pred_test_random)

# Print evaluation metrics after RandomizedSearchCV
print(f"\nTest Metrics (After Randomized Search):")
print(f"Mean Squared Error (MSE): {mse_test_random:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_test_random:.2f}")
print(f"Mean Absolute Error (MAE): {mae_test_random:.2f}")
print(f"R² Score: {r2_test_random:.4f}")

# Save the model
joblib.dump(best_rf_reg, 'rf_model.pkl')
print("Model saved as 'rf_model.pkl'")

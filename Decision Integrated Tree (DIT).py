# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
data = pd.read_csv("Dataset_Emission.csv")

# Split data into features and target variable
X = data.drop(columns=["Pred"])
y = data["Pred"]

data.head()

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Preprocess numerical features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

# Preprocess categorical features
categorical_features = X.select_dtypes(include=['object']).columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
# Combine preprocessing for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])
# Define Decision Integrated Tree model
decision_integrated_tree = Pipeline(steps=[('preprocessor', preprocessor),
                                           ('regressor', GradientBoostingRegressor())])  # You can choose any model here
# Train the model
decision_integrated_tree.fit(X_train, y_train)

# Predict on the test set
y_pred = decision_integrated_tree.predict(X_test)

from sklearn.metrics import r2_score

# Evaluate the model
r_squared = r2_score(y_test, y_pred)
print("R-squared Score:", r_squared)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

from sklearn.metrics import mean_absolute_error

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

print("MAE:", mae)


def get_real_time_data():
    # Your code to collect real-time data goes here
    # For demonstration purposes, let's assume we're collecting data from a sensor
    # This function should return a dictionary containing feature values
    real_time_data = {
        'YYYYMM': '202401',
        'Category': 1
        #other data

    }
    return real_time_data

# Collect real-time data
real_time_data = get_real_time_data()

# Convert real-time data to DataFrame
real_time_df = pd.DataFrame(real_time_data, index=[0])
# Assuming X_train is the DataFrame used during model training
assert set(real_time_df.columns) == set(X_train.columns), "Feature names in real-time data do not match training data"

# Assuming additional_features contains the names of additional features in the real-time data
additional_features = set(real_time_df.columns) - set(X_train.columns)

# Add new transformer steps for additional features
for feature in additional_features:
    if real_time_df[feature].dtype == 'object':
        additional_transformer = ('encoder_' + feature, OneHotEncoder(), [feature])
    else:
        additional_transformer = ('scaler_' + feature, StandardScaler(), [feature])
    preprocessor.transformers.append(additional_transformer)

# Preprocess the real-time data using the updated ColumnTransformer
preprocessed_data = preprocessor.transform(real_time_df)


# Make predictions
predicted_co2_emission = decision_integrated_tree.predict(real_time_df)

print("Predicted CO2 Emission:", predicted_co2_emission)

import matplotlib.pyplot as plt

# Plot actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Actual vs. Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='red', label='Ideal')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted CO2 Emission')
plt.legend()
plt.show()

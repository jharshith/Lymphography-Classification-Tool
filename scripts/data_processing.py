# scripts/data_processing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# Load the .names file to understand the structure
names_file_path = './data/lymphography.names'
with open(names_file_path, 'r') as file:
    names_content = file.read()
# print(names_content)

# Load the .data file into a DataFrame
data_file_path = './data/lymphography.data'
data = pd.read_csv(data_file_path, header=None)

# Assign column names to the DataFrame
column_names = [
    "class", "lymphatics", "block_of_affere", "bl_of_lymph_c", "bl_of_lymph_s", "by_pass", "extravasates",
    "regeneration_of", "early_uptake_in", "lym_nodes_dimin", "lym_nodes_enlar", "changes_in_lym",
    "defect_in_node", "changes_in_node", "changes_in_stru", "special_forms", "dislocation_of",
    "exclusion_of_no", "no_of_nodes_in"
]
data.columns = column_names

# Display the first few rows of the data with column names
print(data.head())

# Separate the features and the target variable
X = data.drop("class", axis=1)
y = data["class"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Display the shapes of the training and testing sets
print(X_train_scaled.shape, X_test_scaled.shape, y_train.shape, y_test.shape)

# Train a Random Forest Classifier with class weights
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

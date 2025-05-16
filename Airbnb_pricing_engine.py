# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Creating the DataFrame and Loading the dataset

df = pd.read_csv("Airbnb_dataset.csv")
print(df.head()) # Prints the first 5 rows of the dataset.

print(df.columns) # Prints all the columns name
# Cleaning the dataset

# Checking missing or null values
print(df.isnull().sum())

df.info() # Gives information about the data

# Converting the object datatype into datetime datatype

df["last review"] = pd.to_datetime(df["last review"], errors = "coerce")

# Handling missing or null values

# Filling null/missing values
df.fillna({"reviews per month" : 0, "last review" : df["last review"].min()}, inplace = True)
# Dropping empty columns entries
df.dropna(subset = ["NAME", "host name", 'price', 'neighbourhood group', 'room type', 'number of reviews', 'review rate number' ], inplace = True)
df.info()
df.isnull().sum()

# Removing unwanted columns
df = df.drop(columns = ["house_rules", "license"], errors = "ignore")
df.head()

# Removing $ sign and converting numbers into floating point numbers
df["price"] = df["price"].replace("[/$,]", "", regex = True).astype(float)
df["service fee"] = df["service fee"]. replace("[/$,]", "", regex = True).astype(float)
print(df.head())

# Removing duplicates
df.drop_duplicates(inplace = True)

# Descriptive Statistics
df.describe()

# Filling missing numeric values with the median.
for col in ["price", "service fee"]:
    df[col] = df[col].fillna(df[col].median())
print(df.isnull().sum())

# Exploratory Data Analysis(EDA)

# Price by Room Type and Neighbourhood
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='room type', y='price', hue='neighbourhood group')
plt.ylim(0, 500)  # Limit y-axis for readability
plt.title("Price by Room Type and Location")
plt.show()

# Correlation heatmap
corr_features = df[['price', 'number of reviews', 'review rate number', 'reviews per month', 'service fee']]
sns.heatmap(corr_features.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Create binary and categorical features
df['is_verified'] = df['host_identity_verified'] == 'verified'
df['season'] = pd.to_datetime(df['last review'], errors='coerce').dt.month % 12 // 3 + 1  # 1=Winter, 2=Spring, etc.

# One-hot encode categorical variables
df_model = pd.get_dummies(df[[
    'price', 'neighbourhood group', 'room type', 'is_verified', 
    'number of reviews', 'review rate number', 'reviews per month', 'season'
]], drop_first=True)

# Regression Modeling to Predict Price
# Defining features and target
X = df_model.drop(columns='price')
y = df_model['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

'''R² Score: -0.00010354173896898189
RMSE: 330.19126701446123'''

# Saving cleaned and model-ready data for Tableau
df.to_csv("Airbnb_Cleaned.csv", index=False)
df_model.to_csv("Airbnb_ModelReady.csv", index=False)

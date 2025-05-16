
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Loading Data
df = pd.read_csv("healthcare_appointments_no_show.csv")

# Checking for Missing Values: Identify and handle missing values.
print(df.isnull().sum())
df.fillna("Unknown", inplace=True)  # Replace missing values
print(df.isnull().sum())

# Converting Data Types
df['Appointment_Date'] = pd.to_datetime(df['Appointment_Date'], format="%d-%m-%Y")
df['Age'] = df['Age'].astype(int)
df['No_Show'] = df['No_Show'].astype(int)

# Feature Engineering: Creating new features like Time_of_Day and Weekend_Appointment.

df['Time_of_Day'] = pd.cut(df['Appointment_Time'], bins=[0, 12, 18, 24], labels=['Morning', 'Afternoon', 'Evening'])
df['Weekend_Appointment'] = df['Day_of_Week'].isin(['Saturday', 'Sunday']).astype(int)

# Train Decision Tree Model to Predict No-Shows
# Split Data: Separating features and target variable.

X = df[['Age', 'Insurance_Status', 'Appointment_Type', 'Day_of_Week', 'Reminder_System', 'Previous_Attendance']]
y = df['No_Show']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encoding Categorical Variables: Converting categorical data into numerical format.

le = LabelEncoder()
for col in ['Insurance_Status', 'Appointment_Type', 'Day_of_Week', 'Reminder_System']:
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

# Training Decision Tree Model:

model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Evaluating Model Performance:

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

df.to_csv("cleaned_healthcare_data.csv", index=False)

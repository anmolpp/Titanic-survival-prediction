# Import Required Libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score




# Load the Titanic Dataset 
data = pd.read_csv(r"C:\Users\parja\Downloads\Titanic-Dataset.csv") 

# Fill missing values
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Convert categorical variables into dummy/indicator variables
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

# Drop unnecessary columns
data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Features and Target
X = data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']]
y = data['Survived']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and Train the Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Create a copy of the test data and add predictions
X_test_copy = X_test.copy()
X_test_copy['Predicted_Survived'] = y_pred
X_test_copy['Actual_Survived'] = y_test.values


# Save to CSV
X_test_copy.to_csv("titanic_predictions.csv", index=False)

# Print the first few rows
print(X_test_copy[['Pclass', 'Age', 'Predicted_Survived', 'Actual_Survived']].head(10))

# Count how many survived vs. not survived in predictions
prediction_counts = X_test_copy['Predicted_Survived'].value_counts()

# Labels and values for the pie chart
labels = ['Did Not Survive', 'Survived']
sizes = [prediction_counts[0], prediction_counts[1]]
colors = ['lightcoral', 'lightskyblue']

# Create the pie chart
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Predicted Survival Distribution')
plt.axis('equal')  # Makes it a circle
plt.show()


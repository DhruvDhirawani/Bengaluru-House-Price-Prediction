# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "G:\Data Science Projects\Bengaluru House Price Prediction\Bengaluru_House_Data.csv"
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe
df.head()

# Display basic information about the dataset
df.info()

# Check for missing values
df.isnull().sum()

# Fill missing values for 'bath' and 'balcony' with median values
df['bath'].fillna(df['bath'].median(), inplace=True)
df['balcony'].fillna(df['balcony'].median(), inplace=True)

# Drop rows with missing 'size' or 'total_sqft' as they are crucial for analysis
df.dropna(subset=['size', 'total_sqft'], inplace=True)

# Extract number of bedrooms from 'size'
df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))

# Function to convert 'total_sqft' to a single number
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)
    except:
        return None

# Apply the conversion function
df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)

# Drop rows where 'total_sqft' could not be converted
df.dropna(subset=['total_sqft'], inplace=True)

# Distribution of house prices
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], bins=50, kde=True)
plt.title('Distribution of House Prices')
plt.xlabel('Price (in lakhs)')
plt.ylabel('Frequency')
plt.show()

# Scatter plot of total_sqft vs price
plt.figure(figsize=(10, 6))
sns.scatterplot(x='total_sqft', y='price', data=df)
plt.title('Total Square Feet vs Price')
plt.xlabel('Total Square Feet')
plt.ylabel('Price (in lakhs)')
plt.show()

# Correlation heatmap
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Importing necessary libraries for modeling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Define features and target variable
X = df[['total_sqft', 'bath', 'balcony', 'bhk']]
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

mae, r2

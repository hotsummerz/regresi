import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("shopping_trends.csv")

# Variabel
data = df[[
    'Age',
    'Review Rating',
    'Previous Purchases',
    'Discount Applied',
    'Purchase Amount (USD)'
]].copy()

# Encoding
data['Discount Applied'] = (
    data['Discount Applied'] == 'Yes'
).astype(int)

# Cek Data
print(data.head())

print(data.isnull().sum())

# Scatter Plot
sns.scatterplot(
    x='Previous Purchases',
    y='Purchase Amount (USD)',
    data=data
)

plt.title("Previous Purchases vs Purchase Amount")

plt.show()

# Boxplot
sns.boxplot(
    y=data['Purchase Amount (USD)']
)

plt.title("Boxplot Purchase Amount")

plt.show()

# Heatmap
plt.figure(figsize=(10,6))

sns.heatmap(
    data.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.xticks(rotation=30, ha='right')

plt.tight_layout()

plt.show()

# Regresi
X = data[[
    'Age',
    'Review Rating',
    'Previous Purchases',
    'Discount Applied'
]]

y = data['Purchase Amount (USD)']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluasi
print("R2 Score:", r2_score(y_test, y_pred))


# Koefisien
coef = pd.DataFrame({
    'Variabel': X.columns,
    'Koefisien': model.coef_
})

print(coef)
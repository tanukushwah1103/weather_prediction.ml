import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = {
    'Humidity': [80, 60, 70, 90, 85, 65, 75, 78, 88, 55],
    'WindSpeed': [12, 7, 10, 5, 8, 9, 6, 11, 4, 13],
    'Temperature': [30, 25, 28, 32, 31, 26, 29, 27, 33, 24]
}
df = pd.DataFrame(data)

X = df[['Humidity', 'WindSpeed']]
y = df['Temperature']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.scatter(y_test, y_pred, color='purple')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linestyle='--')
plt.xlabel('Actual Temperature')
plt.ylabel('Predicted Temperature')
plt.title('Actual vs Predicted Temperature')
plt.grid(True)
plt.savefig('output_graph.png')

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import r2_score

data = {
    "years_experience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "salary": [25000, 27000, 30000, 31500, 34000, 36000, 39000, 41000, 44000, 47000]
}

df = pd.DataFrame(data)
#print(df)



# X måste vara 2D (antal rader, antal features); y kan vara 1D
X = df[["years_experience"]]   # 2D
y = df["salary"]               # 1D

model = LinearRegression()
model.fit(X, y)

slope = model.coef_[0]       # lutning (kr per extra år)
intercept = model.intercept_ # skärning (kr vid 0 år)
print("Lutning (kr/år):", slope)
print("Intercept (kr):", intercept)

x_line = np.linspace(df["years_experience"].min(), df["years_experience"].max(), 100).reshape(-1, 1)
y_line = model.predict(x_line)

plt.figure()
plt.scatter(df["years_experience"], df["salary"], label="Data")
plt.plot(x_line, y_line, label="Linjär regression")
plt.xlabel("År i arbete")
plt.ylabel("Lön (kr/månad)")
plt.title("Linjär modell: lön = intercept + lutning * år")
plt.legend()
plt.show()

y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
print("R^2:", r2)

years_new = pd.DataFrame({"years_experience": [0, 5, 12]})
predictions = model.predict(years_new)

for yrs, sal in zip(years_new["years_experience"], predictions):
    print(f"Prognos vid {yrs} år: {sal:.0f} kr/månad")
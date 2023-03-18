# Module Import
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Data import
file_url = 'https://media.githubusercontent.com/media/musthave-ML10/data_source/main/insurance.csv'
data = pd.read_csv(file_url)

# 2. Dataset Check
# print(round(data.describe(), 2))

# 3. Data Preprocessing
# 우리는 나이, 성별, bmi, 자녀 수, 흡연여부등을 통해 보혐료를 예측하는 문제를 해결할 것이다.
X = data[['age', 'sex', 'bmi', 'children', 'smoker']] # 독립변수 : 나이, 성별, bmi, 자녀 수, 흡연여부
Y = data['charges'] # 종속변수 : 보험료

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=100)

# print(f"x_train ▼ \n {x_train}")
# print(f"x_test ▼ \n {x_test}")
# print()
# print(f"y_train ▼ \n {y_train}")
# print(f"y_test ▼ \n {y_test}")

# 4. Modeling
model = LinearRegression()
model.fit(x_train, y_train)

# 5. Prediction
pred = model.predict(x_test)

# 6-1. Visualization
# comparison = pd.DataFrame({'actual': y_test, 'pred': pred})
# plt.figure(figsize=(10,10)) # 그래프 크기
# sns.scatterplot(x='actual', y='pred', data=comparison)
# plt.show()

#6-2. RMSE
mse = mean_squared_error(y_test, pred)
print(mse)

rmse = mean_squared_error(y_test, pred, squared=False)
print(rmse)

print(model.score(x_train, y_train))
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from matplotlib import pyplot as plt

# 读取数据
df = pd.read_csv('new_data_percent.csv', encoding='ISO-8859-1')

# 相关性分析
df_tmp1 = df[['Location', 'QuadraticTerm', 'LinearTerm', 'DF', 'Endingpoint', 'Tvalue_min', 'Duration', 'Mean_F0', 'T3_percent']]
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.heatmap(df_tmp1.corr(), cmap='YlGnBu', annot=True, linewidth=.5)
plt.title('Variable Correlation')
plt.show()

# 划分自变量和目标变量
X = df.iloc[:, 11:19]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=42)

# 设置KNN模型的参数
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]  # p=1 for Manhattan distance, p=2 for Euclidean distance
}

# 网格搜索法确定最佳参数
knn = KNeighborsRegressor()
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("Best Parameters:", grid_search.best_params_)
print("Best MSE Score (Negative):", grid_search.best_score_)

# 使用最佳模型进行预测
best_knn = grid_search.best_estimator_
predictions = best_knn.predict(X_test)

# 计算 MSE
best_mse = mean_squared_error(y_test, predictions)
print('MSE of the best model: ', best_mse)

# 计算 RMSE
best_rmse = np.sqrt(best_mse)
print('RMSE of the best model: ', best_rmse)

# 计算 MAE
best_mae = mean_absolute_error(y_test, predictions)
print('MAE of the best model: ', best_mae)

# 计算 R²
best_r2 = r2_score(y_test, predictions)
print('R² of the best model: ', best_r2)

# 预测结果可视化 (前 100 个样本)
plt.figure()
plt.plot(np.arange(100), y_test[:100], "go-", label="True value")
plt.plot(np.arange(100), predictions[:100], "ro-", label="Predict value")
plt.title("Predicting Outcomes (First 100 Samples)")
plt.legend(loc="best")
plt.show()


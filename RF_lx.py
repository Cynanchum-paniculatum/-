# -*- coding = utf-8 -*-

import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from matplotlib import pyplot as plt

# 读取数据
df = pd.read_csv('new_data_percent.csv', encoding='ISO-8859-1')

# 数据相关性分析
df_tmp1 = df[['Location', 'QuadraticTerm','LinearTerm','DF', 'Endingpoint', 'Tvalue_min', 'Duration', 'Mean_F0', 'T3_percent']]
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.heatmap(df_tmp1.corr(), cmap = 'YlGnBu', annot = True, linewidth=.5)
plt.title('Variable Correlation')
plt.show()

# 划分自变量和目标变量
X = df.iloc[:, 11:19]
y = df.iloc[:, -1]

# 将数据集拆分为70%训练样本数据，30%为测试样本数据
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=42)

# 设置随机森林回归模型的参数
param_grid = {
    'n_estimators': [50, 100, 200, 300, 400],
    'max_depth': [None, 5, 10, 20, 50],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 3, 5],
    'max_features': ['log2', 'sqrt']
}

# 网格搜索法确定最佳参数
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# 输出最佳参数
print(grid_search.best_params_)
print(grid_search.best_score_)

# 可视化处理
results = pd.DataFrame(grid_search.cv_results_)
mse = abs(results['mean_test_score'])

plt.figure(figsize=(10, 8))
plt.title("Random Forest Grid Search MSEs", fontsize=16)
plt.xlabel("Parameter Set", fontsize=16)
plt.ylabel("MSE Score", fontsize=16)
plt.plot(range(1, len(mse) + 1), mse, color='#d7191c', linestyle='--', label='MSE')
plt.grid(False)
plt.legend(loc='best', fontsize=14)
plt.show()
plt.clf()

# 输出最佳模型的MSE、RMSE、MAE和R²
best_rf = grid_search.best_estimator_
predictions = best_rf.predict(X_test)

# 打印预测值和真实值，确保模型运行正常
print("Predictions: ", predictions[:10])
print("True values: ", y_test[:10])

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

# 预测结果可视化
plt.figure()
plt.plot(np.arange(100), y_test[:100], "go-", label="True value")
plt.plot(np.arange(100), predictions[:100], "ro-", label="Predict value")
plt.title("Predicting Outcomes")
plt.legend(loc="best")
plt.show()

# 设置马克龙色调
colors = ['#FCE38A', '#EAFFD0', '#95E1D3', '#F38181', '#dfc27d', '#fbb4b9', '#7fcdbb', '#377eb8', '#abdda4']

# 可视化特征重要性
feature_importances = best_rf.feature_importances_
feature_names = X.columns
sorted_idx = feature_importances.argsort()

plt.figure(figsize=(10, 8))
plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], color=colors)
plt.yticks(range(len(sorted_idx)), feature_names[sorted_idx], fontsize=12)
plt.xlabel("Feature Importance", fontsize=16)
plt.title("Random Forest Feature Importance", fontsize=16)

# 添加数值标签
for i, v in enumerate(feature_importances[sorted_idx]):
    plt.text(v + 0.001, i, f"{v:.3f}", fontsize=12)
plt.tight_layout()
plt.show()

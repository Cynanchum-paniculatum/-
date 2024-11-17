import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt

# 读取数据
df = pd.read_csv('new_data_percent.csv', encoding='ISO-8859-1')

# 相关性分析
df_tmp1 = df[['Location','QuadraticTerm','LinearTerm', 'DF', 'Endingpoint', 'Tvalue_min', 'Duration', 'Mean_F0', 'T3_percent']]
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.heatmap(df_tmp1.corr(), cmap='YlGnBu', annot=True, linewidth=.5)
plt.title('Variable Correlation')
plt.show()

# 划分自变量和目标变量
X = df.iloc[:, 11:19]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=42)

# 设置较小的参数搜索空间
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.1],
    'epsilon': [0.1, 0.2],
    'kernel': ['rbf']  # 只使用RBF核
}

# 网格搜索法确定最佳参数
svr = SVR()
grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("Best Parameters:", grid_search.best_params_)
print("Best MSE Score:", grid_search.best_score_)

# 输出最佳模型的MSE、RMSE、MAE和R²
best_svr = grid_search.best_estimator_
predictions = best_svr.predict(X_test)

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

# 计算自变量重要性 (Permutation Importance)
importance_result = permutation_importance(best_svr, X_test, y_test, n_repeats=10, random_state=42)
importance_scores = importance_result.importances_mean

# 用马卡龙色调可视化自变量重要性
macaron_colors = ['#FFB7C5', '#FFDAC1', '#E2F0CB', '#B5EAD7', '#C7CEEA', '#FF9AA2', '#FFB347', '#FFD700', '#AEC6CF']

# 排序并可视化
sorted_idx = np.argsort(importance_scores)
plt.figure(figsize=(10, 6))
plt.barh(X.columns[sorted_idx], importance_scores[sorted_idx], color=macaron_colors[:len(sorted_idx)])
plt.xlabel("Importance")
plt.title("Feature Importance (Permutation Importance - Macaron Colors)")
plt.show()

# 预测结果可视化 (展示前100个样本)
plt.figure()
plt.plot(np.arange(100), y_test[:100], "go-", label="True value")
plt.plot(np.arange(100), predictions[:100], "ro-", label="Predict value")
plt.title("Predicting Outcomes (First 100 Samples)")
plt.legend(loc="best")
plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# 读取数据
df = pd.read_csv('new_Tone_data_assign.csv', encoding='ISO-8859-1')

# 划分自变量和目标变量
X = df.iloc[:, 11:19]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=42)

# 定义模型参数
param_grid_rf = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
param_grid_knn = {'n_neighbors': [3, 5, 7, 10], 'weights': ['uniform', 'distance']}
param_grid_svc = {'C': [0.1, 1, 10], 'gamma': [0.1, 0.01], 'kernel': ['rbf', 'linear']}

# RandomForest 模型
grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=42, class_weight='balanced'), param_grid=param_grid_rf, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search_rf.fit(X_train, y_train)
best_rf = grid_search_rf.best_estimator_

# KNN 模型
# 对KNN进行分析时，进行特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X_scaled, y, train_size=0.7, test_size=0.3, random_state=42)

# KNN模型训练
grid_search_knn = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid_knn, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search_knn.fit(X_train_knn, y_train_knn)
best_knn = grid_search_knn.best_estimator_

# SVM 模型
grid_search_svc = GridSearchCV(estimator=SVC(probability=True, random_state=42), param_grid=param_grid_svc, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search_svc.fit(X_train, y_train)
best_svc = grid_search_svc.best_estimator_

# 获取模型的预测概率
y_prob_rf = best_rf.predict_proba(X_test)[:, 1]
y_prob_knn = best_knn.predict_proba(X_test_knn)[:, 1]
y_prob_svc = best_svc.predict_proba(X_test)[:, 1]

# 计算每个模型的ROC曲线和AUC值
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

fpr_knn, tpr_knn, _ = roc_curve(y_test_knn, y_prob_knn)
roc_auc_knn = auc(fpr_knn, tpr_knn)

fpr_svc, tpr_svc, _ = roc_curve(y_test, y_prob_svc)
roc_auc_svc = auc(fpr_svc, tpr_svc)

# 绘制ROC曲线
plt.figure(figsize=(8, 6))

macaron_colors = ['#ADD8E6', '#FF7F50', '#6A5ACD', '#2E8B57', '#87CEFA']

plt.plot(fpr_rf, tpr_rf, color=macaron_colors[0], lw=2, label=f'RandomForest (AUC = {roc_auc_rf:.2f})')
plt.plot(fpr_knn, tpr_knn, color=macaron_colors[1], lw=2, label=f'KNN (AUC = {roc_auc_knn:.2f})')
plt.plot(fpr_svc, tpr_svc, color=macaron_colors[2], lw=2, label=f'SVM (AUC = {roc_auc_svc:.2f})')

plt.plot([0, 1], [0, 1], color=macaron_colors[3], linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for SVM, KNN, and RandomForest (Macaron Colors)')
plt.legend(loc="lower right")
plt.show()

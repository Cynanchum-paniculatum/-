import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef, cohen_kappa_score
from matplotlib import pyplot as plt

# 读取数据
df = pd.read_csv('learner_finally.csv', encoding='ISO-8859-1')

# 相关性分析
df_tmp1 = df[
    ['Location', 'QuadraticTerm','LinearTerm','DF', 'Endingpoint', 'Tvalue_min', 'Duration',
     'Mean_F0', 'Length', 'HSKlevel', 'Gender', 'Assign']
]
sns.heatmap(df_tmp1.corr(), cmap='YlGnBu', annot=True, linewidth=.5)
plt.title('Variable Correlation')
plt.show()

# 划分自变量和目标变量
X = df.iloc[:, 10:21]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=42)

# 设置SVM模型的参数，减少搜索空间以加快速度
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.1, 0.01],
    'kernel': ['rbf', 'linear']
}

# 网格搜索法确定最佳参数
svc = SVC(random_state=42)
grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("Best Parameters: ", grid_search.best_params_)
print("Best Cross-Validation Score: ", grid_search.best_score_)

# 可视化处理
results = pd.DataFrame(grid_search.cv_results_)
accuracy = results['mean_test_score']

plt.figure(figsize=(10, 8))
plt.title("SVM Grid Search Accuracy", fontsize=16)
plt.xlabel("Parameter Set", fontsize=16)
plt.ylabel("Accuracy Score", fontsize=16)
plt.plot(range(1, len(accuracy) + 1), accuracy, color='#d7191c', linestyle='--', label='Accuracy')
plt.grid(False)
plt.legend(loc='best', fontsize=14)
plt.show()

# 输出最佳模型的Accuracy和分类报告
best_svc = grid_search.best_estimator_
predictions = best_svc.predict(X_test)
best_accuracy = accuracy_score(y_test, predictions)
print('Accuracy of the best model: ', best_accuracy)
print('Classification Report:\n', classification_report(y_test, predictions))

# 计算Precision, Recall, F1 Score, MCC, OA, Kappa
precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')
mcc = matthews_corrcoef(y_test, predictions)
oa = accuracy_score(y_test, predictions)
kappa = cohen_kappa_score(y_test, predictions)

print('Precision: ', precision)
print('Recall: ', recall)
print('F1 Score: ', f1)
print('Matthews Correlation Coefficient (MCC): ', mcc)
print('Overall Accuracy (OA): ', oa)
print('Cohen\'s Kappa: ', kappa)

# 混淆矩阵可视化
conf_matrix = confusion_matrix(y_test, predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix", fontsize=16)
plt.xlabel("Predicted Label", fontsize=14)
plt.ylabel("True Label", fontsize=14)
plt.show()

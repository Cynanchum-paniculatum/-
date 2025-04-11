import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef, cohen_kappa_score, roc_curve, auc
from matplotlib import pyplot as plt

# 读取数据
df = pd.read_csv('new_Tone_data_assign.csv', encoding='ISO-8859-1')

# 相关性分析
df_tmp1 = df[['Location', 'QuadraticTerm', 'LinearTerm', 'DF', 'Endingpoint', 'Tvalue_min', 'Duration', 'Mean_F0', 'Assign']]
sns.heatmap(df_tmp1.corr(), cmap='YlGnBu', annot=True, linewidth=.5)
plt.title('Variable Correlation')
plt.show()

# 划分自变量和目标变量
X = df.iloc[:, 11:19]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=42)

# 设置Random Forest模型的参数，适度扩展超参数范围并处理类别不平衡问题
param_grid = {
    'n_estimators': [50, 100, 200],  # 适度扩展树的数量
    'max_depth': [None, 10, 20, 30],  # 适度增加深度
    'min_samples_split': [2, 5, 10],  # 保持相同的分裂标准
    'min_samples_leaf': [1, 2, 4],# 保持叶节点最小样本数的选择
    'max_features': ['auto', 'sqrt', 'log2']  # 添加 max_features 参数
}

# 使用class_weight='balanced' 来处理类别不平衡问题
rf = RandomForestClassifier(random_state=42, class_weight='balanced')

# 使用GridSearchCV进行网格搜索，scoring依然为accuracy
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("Best Parameters: ", grid_search.best_params_)
print("Best Cross-Validation Score: ", grid_search.best_score_)

# 可视化处理
results = pd.DataFrame(grid_search.cv_results_)
accuracy = results['mean_test_score']

plt.figure(figsize=(10, 8))
plt.title("Random Forest Grid Search Accuracy", fontsize=16)
plt.xlabel("Parameter Set", fontsize=16)
plt.ylabel("Accuracy Score", fontsize=16)
plt.plot(range(1, len(accuracy) + 1), accuracy, color='#d7191c', linestyle='--', label='Accuracy')
plt.grid(False)
plt.legend(loc='best', fontsize=14)
plt.show()

# 使用最佳模型进行预测
best_rf = grid_search.best_estimator_
predictions = best_rf.predict(X_test)

# 输出最佳模型的Accuracy和分类报告
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

# 计算特征重要性
feature_importance = best_rf.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# 颜色列表
colors = ['#FCE38A', '#EAFFD0', '#95E1D3', '#F38181', '#dfc27d', '#fbb4b9', '#7fcdbb', '#abdda4']

# 创建条形图并应用自定义颜色
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette=colors[:len(feature_importance_df)])

plt.title('Feature Importance from Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')

# 添加数值标签
for index, value in enumerate(feature_importance_df['Importance']):
    plt.text(value, index, f'{value:.3f}', va='center')

plt.show()

# 预测结果可视化 (展示前100个样本)
plt.figure()
plt.plot(np.arange(100), y_test[:100], "go--", label="True value")
plt.plot(np.arange(100), predictions[:100], "ro-", label="Predicted value")
plt.title("True vs Predicted Labels (First 100 Samples)")
plt.legend(loc="best")
plt.show()

# 计算ROC曲线和AUC值
y_prob = best_rf.predict_proba(X_test)[:, 1]  # 获取正类的预测概率
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

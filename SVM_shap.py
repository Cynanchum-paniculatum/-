# 导入调色板
import shap
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
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

# 设置SVM模型的参数，减少搜索空间以加快速度
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.1, 0.01],
    'kernel': ['rbf', 'linear']
}

# 网格搜索法确定最佳参数
svc = SVC(random_state=42, probability=True)  # 必须设置probability=True
grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("Best Parameters: ", grid_search.best_params_)
print("Best Cross-Validation Score: ", grid_search.best_score_)

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

# 绘制ROC曲线
y_prob = best_svc.predict_proba(X_test)[:, 1]  # 获取正类的预测概率
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

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

# 使用SHAP进行解释
explainer = shap.Explainer(best_svc, X_train)  # 使用shap.Explainer
shap_values = explainer(X_test)

# 绘制Summary Plot，显示所有特征
shap.summary_plot(shap_values, X_test, feature_names=X.columns)

# 计算每个特征的平均绝对值 SHAP 值
shap_abs_values = np.abs(shap_values.values).mean(axis=0)

# 创建一个 DataFrame，存储特征名称和对应的平均绝对 SHAP 值
shap_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Mean Absolute SHAP Value': shap_abs_values
})

# 根据 SHAP 值排序，方便查看重要性
shap_importance_df = shap_importance_df.sort_values(by='Mean Absolute SHAP Value', ascending=False)

# 输出特征的平均绝对 SHAP 值
print(shap_importance_df)

# 设置自定义多种颜色的调色板 (马卡龙风格或其他自定义颜色)
macaron_colors = ["#FFB6C1", "#FFDAB9", "#E6E6FA", "#F0E68C", "#ADD8E6", "#98FB98", "#FFDEAD", "#D8BFD8", "#FFE4E1"]

# 如果需要图表展示
plt.figure(figsize=(10, 6))

# 使用自定义的颜色列表来绘制条形图
sns.barplot(x='Mean Absolute SHAP Value', y='Feature', data=shap_importance_df, palette=macaron_colors)

plt.title('Mean Absolute SHAP Values (Feature Importance)', fontsize=16)
plt.xlabel('Mean Absolute SHAP Value', fontsize=14)
plt.ylabel('Feature', fontsize=14)

# 显示图形
plt.show()

# 预测结果可视化 (展示前100个样本)
plt.figure()
plt.plot(np.arange(100), y_test[:100], "go-", label="True value")
plt.plot(np.arange(100), predictions[:100], "ro-", label="Predicted value")
plt.title("True vs Predicted Labels (First 100 Samples)")
plt.legend(loc="best")
plt.show()


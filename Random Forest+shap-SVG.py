# 导入需要用到的Python库
import shap
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

# 这个设置是为了让画图时支持中文
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 直接读取本地的.csv数据集来得到 X, y
df = pd.read_csv('桥下-归一B.csv')
X, y = df.drop('Y', axis=1), df['Y'].values

X = X.rename(columns={'Road': 'S-R',
                     'Fence': 'S-F',
                     'Plant': 'S&G-P',
                     'Tree': 'G-T',
                     'Understory planting': 'G-U',
                     'Ground': 'G&V-G',
                     'Furniture': 'V-F',
                     'Wall': 'V-W',
                     'Sidewalk': 'V&S-S'})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                    random_state = 10)

print('X_train数据的维度为: ', X_train.shape)
print('X_test数据的维度为: ', X_test.shape)

# In[]: 解释Random Forest模型

model = RandomForestRegressor(random_state=10) # 固定 random state，每次运行的结果才会是一样的。
model.fit(X_train, y_train)
y_train_hat = model.predict(X_train)
y_test_hat = model.predict(X_test)

print('训练集上的R2:', r2_score(y_train, y_train_hat))
print('测试集上的R2:', r2_score(y_test, y_test_hat))

explainer = shap.Explainer(model)
shap_values = explainer(X_test)
shap_values.shape

# 我们可以改变散点图、蜜蜂图、热力图的配色，选色网站：
# https://matplotlib.org/stable/users/explain/colors/colormaps.html

# 1 瀑布图
sample_ind = 0
print('- 该样本各个特征变量的值为:\n', X_test.iloc[sample_ind])
print('- 该样本各个特征变量的shap值:',
      shap_values.values[sample_ind,:])
print('- 该样本得到的模拟值为:', y_test_hat[sample_ind])
fig = plt.figure()
shap.plots.waterfall(shap_values[sample_ind], max_display=14)

# 2 力图
shap.plots.force(shap_values[sample_ind], matplotlib=True)

# 3 散点图
# fig = plt.figure()
# shap.plots.scatter(shap_values[:, "A1"], color=shap_values) # 用原始配色
# shap.plots.scatter(shap_values[:, "A1"], color=shap_values, cmap=plt.get_cmap("plasma")) # 改变配色

shap.plots.scatter(shap_values[:, "Plant"], color=model.predict(X_test) )  #用模型预测值着色

shap.plots.scatter(shap_values[:, "Plant"], color=shap_values[:, "Tree"])   #A1对于A2的着色

# 4 蜜蜂图
fig = plt.figure()
shap.plots.beeswarm(shap_values) # 用原始配色
# shap.plots.beeswarm(shap_values, color=plt.get_cmap("plasma")) # 改变配色

# 5 热力图
fig = plt.figure()
shap.plots.heatmap(shap_values) # 用原始配色
# shap.plots.heatmap(shap_values, cmap=plt.get_cmap("seismic")) # 改变配色

# 6 柱状图
print('各个特征变量shap值绝对值的平均:',
      np.nanmean(abs(shap_values.values), axis=0))
fig = plt.figure()
shap.plots.bar(shap_values)

#计算 SHAP 交互值
# shap_interaction_values = explainer.shap_interaction_values(X_test)
# 绘制 SHAP 特征交互图
#shap.summary_plot(shap_interaction_values, X_test)
# 显示图形
# plt.show()


# 保存SHAP数据
sample_shap_values = shap_values.values[sample_ind, :]
simulated_value = y_test_hat[sample_ind]
shap_abs_mean = np.nanmean(abs(shap_values.values), axis=0)
# 创建一个 DataFrame 来保存数据
feature_names = X_test.columns
data = {
    '特征名称': feature_names,
    '样本 SHAP 值': sample_shap_values,
    'SHAP 值绝对值的平均': shap_abs_mean
}
df_shap_data = pd.DataFrame(data)
column_order = ['特征名称', '样本 SHAP 值', 'SHAP 值绝对值的平均']
df_shap_data = df_shap_data[column_order]
new_row = pd.DataFrame({'特征名称': ['模拟值'], '样本 SHAP 值': [simulated_value], 'SHAP 值绝对值的平均': [None]})
new_row = new_row.astype({'特征名称': 'object', '样本 SHAP 值': df_shap_data['样本 SHAP 值'].dtype, 'SHAP 值绝对值的平均': df_shap_data['SHAP 值绝对值的平均'].dtype})
df_shap_data = pd.concat([df_shap_data, new_row], ignore_index=True)
df_shap_data.to_csv('shap_data.csv', index=False, encoding='utf-8-sig')



#评价模型
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns

# 计算预测值
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 1. 核心指标计算
print("\n=== 模型评价指标 ===")
print("训练集:")
print(f"R²: {r2_score(y_train, y_train_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_train, y_train_pred):.4f}")
print(f"MSE: {mean_squared_error(y_train, y_train_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.4f}")

print("\n测试集:")
print(f"R²: {r2_score(y_test, y_test_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_test_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_test_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")

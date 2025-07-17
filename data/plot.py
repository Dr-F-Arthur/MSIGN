import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
'''
# 读取CSV文件数据
train = pd.read_csv('cb1_ki.csv')
data2 = pd.read_csv('cb1_ic50.csv')
# 设置图表风格为白色背景
sns.set_style("white")

# 设置颜色样式为'deep'
sns.set_color_codes(palette='deep')

# 创建一个大小为(8, 7)的图表
f, ax = plt.subplots(figsize=(8, 7))

# 绘制训练数据集中'SalePrice'的分布图，并使用蓝色作为颜色
sns.distplot(train['Gbinding Average'], color="green")

# 关闭X轴网格线
ax.xaxis.grid(False)

# 设置Y轴标签为"Frequency"
ax.set(ylabel="Frequency")

# 设置X轴标签为"SalePrice"
ax.set(xlabel="affinity")

# 设置图表标题为"SalePrice distribution"
ax.set(title="PDBBind Affinities distribution")

# 删除图表上和数据轴无关的边框
sns.despine(trim=True, left=True)

# 显示图表
plt.savefig('mixlable_distribution.jpg', dpi=400)
'''
'''

# 读取数据
df_ki = pd.read_csv('cb1_ki.csv')
df_ic50 = pd.read_csv('cb1_ic50.csv')
df_mix = pd.read_csv('cb1_ligand_final.csv')


# 提取并清洗数据
dis_ki = df_ki['Gbinding Average'].dropna()
dis_ic50 = df_ic50['Gbinding Average'].dropna()
dis_mix = df_mix['Gbinding Average'].dropna()


# 设置图形样式
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")

# 绘制KDE图
sns.kdeplot(dis_ki, label='Ki', color='#69A7D1', fill=True, alpha=0.3)
sns.kdeplot(dis_ic50, label='IC50', color='#ECB17C', fill=True, alpha=0.3)
sns.kdeplot(dis_mix, label='Mix', color='#6AC46A', fill=True, alpha=0.3)


ax = plt.gca()  # 获取当前坐标轴对象
ax.xaxis.grid(False)  # 关闭X轴网格线（保留Y轴网格线）

# plt.boxplot([dis_ki, gbind_ic50], labels=['cb1_ki', 'cb1_ic50'])
# 添加标签和标题
plt.xlabel('Affinity', fontsize=16)
plt.ylabel('Density', fontsize=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# plt.title('Distribution Comparison of Affinities', fontsize=14)
plt.legend(fontsize=16)

# 显示图形
plt.savefig('mixlabel_distribution.jpg', dpi=600)
'''
# '''
# import matplotlib as mpl
#
# mpl.rcParams['axes.linewidth'] = 3
# 添加数据集标识列
df_ki = pd.read_csv('cb1_ki.csv')
# df_ki = df_ki['Gbinding Average'].dropna()
df_ic50 = pd.read_csv('cb1_ic50.csv')
# df_ic50 = df_ic50['Gbinding Average'].dropna()
df_mix = pd.read_csv('cb1_ligand_final.csv')
df_ki['Dataset'] = 'Ki'
df_ic50['Dataset'] = 'IC50'
df_mix['Dataset'] = 'Mix'

# 合并数据框
combined_df = pd.concat([df_ki, df_ic50, df_mix], ignore_index=True)

plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")

# 创建箱线图
ax = sns.boxplot(
    x='Dataset',
    y='Gbinding Average',
    data=combined_df,
    palette=['#69A7D1', '#ECB17C', '#6AC46A'],  # 自定义颜色
    showfliers=True,  # 显示异常值
    flierprops=dict(  # 异常点样式
        marker='o',
        markerfacecolor='r',
        markersize=6,
        markeredgecolor='black'
    ),
    linewidth=2,
    width=0.4  # 箱体宽度
)

# 标注中位数
medians = combined_df.groupby('Dataset')['Gbinding Average'].median()
for i, dataset in enumerate(['Ki', 'IC50', 'Mix']):
    ax.text(i, medians[dataset], f'Median: {medians[dataset]:.2f}',
            ha='center', va='bottom', fontsize=12, color='black')

# 美化图形
# plt.title('Comparison of Gbinding Average Distributions', fontsize=14, pad=20)
plt.xlabel('')
plt.ylabel('Affinities', fontsize=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# sns.despine(trim=True)  # 移除上/右边框线

plt.savefig('1.jpg', dpi=600)

# '''
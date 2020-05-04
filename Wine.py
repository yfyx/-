import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from sklearn import datasets, linear_model
from statsmodels.formula.api import ols

# import statsmodels.api as sm
# 读取数据
df = pd.read_csv('winemag-data_first150k.csv', low_memory=False)
# 数值属性
label_num = ['points', 'price']
label_nom = ['country', 'description', 'designation',
             'province', 'region_1', 'region_2', 'variety', 'winery']
df_num = df[label_num]
df_nom = df[label_nom]


# 数据摘要-----------
# 属性可能取值频数
def frequency(label):
    f = df[label].value_counts()
    f = pd.DataFrame(f)
    return f


# 标称属性频数输出
def OutNominal():
    data = open("标称属性频数.txt", 'w+',encoding='utf-8')
    for item in lable_nom:
        print("{}可能取值:\n{}".format(item, frequency(item)),file=data)
    data.close()

#数值属性5数概括
def Num5():
    for item in label_num:
        Minimum = df[item].min()
        Maximum = df[item].max()
        Q1 = df[item].quantile(0.25)
        Median = df[item].mean()
        Q3 = df[item].quantile(0.75)
        print("{}五数概括为：{}，{}，{}，{}，{}".format(item,Minimum,Q1,Median,Q3,Maximum))
#数值属性缺失值个数
def lostdata(nums, item):
    nulltotal = nums[item].isnull().sum()
    print("{}数据缺失值个数为：{}".format(item, nulltotal))




#数据可视化
#直方图
def plothist(nums):
    plt.hist(x=nums,  # 指定绘图数据
             bins=500,# 指定直方图中条块的个数
             edgecolor = 'black') # 指定直方图的边框色
    # plt.xlabel('points')
    plt.xlabel('price')
    plt.ylabel('频数')
    plt.xlim((1, 200))
    plt.show()
#盒图
def Box(nums):
    l = pd.DataFrame(nums)
    l.boxplot()
    plt.show()

def doublenum(nums1,nums2):
    plt.figure(figsize=(30, 14), dpi=98)
    p1=plt.subplot(1,2,1)
    plt.sca(p1)
    nums1.hist(bins=500,edgecolor = 'black')
    plt.xlim((1, 200))
    p2=plt.subplot(1,2,2)
    p1.set_title("填补前",fontsize=18)
    plt.sca(p2)
    nums2.hist(bins=500,edgecolor = 'black')
    plt.xlim((1, 200))
    p2.set_title("填补后",fontsize=18)
    plt.show()
#缺失值处理
#将缺失部分剔除
def delMiss(nums):
    nums=nums.dropna()
    plt.show()

#用最高频率值来填补缺失值
def ModeFill(nums):
    modefill=nums.mode()
    modedata = modefill.iloc[0]  # 众数值提取
    numsf = nums.fillna(modedata)  # 填充
    doublenum(nums,numsf)

#通过属性的相关关系来填补缺失值
def corrsp(nums):
    cs = nums.corr()
    print(cs)
    return
def corfill(nums):
    plt.figure(figsize=(30, 14), dpi=98)
    p1 = plt.subplot(1, 2, 1)
    p2 = plt.subplot(1, 2, 2)
    plt.sca(p1)
    nums['price'].hist(bins=500, edgecolor='black')
    plt.xlim((1, 200))
    plt.set_title("填补前", fontsize=18)
    wine_model = ols("price ~points",data=nums).fit()
    price_pred = wine_model.predict(nums['points'])
    for i in range(len(nums['price'])):
        if (np.isnan(nums['price'][i])):
            nums.loc[i, 'price']  = int(price_pred[i])
    plt.sca(p2)
    nums['price'].hist(bins=500, edgecolor='black')
    plt.xlim((1, 200))
    p2.set_title("填补后", fontsize=18)
    plt.show()

#通过属性相似性来填补缺失值
df_sim = df[['price','points']]
pp = {}
for row in df_sim.iterrows():
    if pp.get(row[1]['points'], None):
        if not np.isnan(row[1]['price']):
            pp[row[1]['points']][0] += row[1]['price']
            pp[row[1]['points']][1] += 1
    else:
        if not np.isnan(row[1]['price']):
            pp[row[1]['points']] = [row[1]['price'], 1]
for k in pp.keys():
    pp[k][0] = round(pp[k][0] / pp[k][1], 4)
for i in range(len(df_sim['price'])):
    if (np.isnan(df_sim['price'][i])):
        df_sim.loc[i, 'price']  = pp[df_sim.loc[i, 'points']][0]
plothist(df_sim['price'])


#OutNominal()
# Num5()
# plothist(df['points'])
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# for item in label_nom:
#     lostdata(df, item)
# plothist(df['price'])
# Box(df['points'])
#Box(df['price'])
# delMiss(df,'price')
#ModeFill(df['price'])
#corrsp(df)
#corfill(df,'price','points')


import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
import json
from sklearn import datasets, linear_model
from statsmodels.formula.api import ols
import json
# 读取数据
df = pd.read_csv('C:/Users/74024/Documents/study/Going/DM/DM427/youtube-new/USvideos.csv', low_memory=False)
with open('C:/Users/74024/Documents/study/Going/DM/DM427/youtube-new/US_category_id.json', 'r') as f:
    j= json.load(f)
# 数值属性
label_num = ['views','likes','dislikes','comment_count']
lable_nom = ['category_id','trending_date','title', 'channel_title', 'tags',
             'comments_disabled', 'ratings_disabled', 'video_error_or_removed',
             'publish_time','description']
df_num = df[label_num]
df_nom = df[lable_nom]


# 数据摘要-----------
# 属性可能取值频数
def frequency(label):
    f = df[label].value_counts()
    f = pd.DataFrame(f)
    return f

# freq = df['trending_date'].value_counts()
# freq
f=frequency('comments_disabled')
# print(f)
def perpie(label):
    plt.figure(figsize = (5, 5))
    plt.pie(np.array(df[label].value_counts()), autopct='%.2f%%', labels = ['False', 'True'])
    plt.xlabel(label+'percentage')
    plt.show()
# perpie('ratings_disabled')
# 标称属性频数输出
def OutNominal():
    for item in lable_nom:
        print("{}可能取值:\n{}".format(item, frequency(item)))

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
def lostdata(nums):
    for item in nums:
         nulltotal = nums[item].isnull().sum()
         print("{}数据缺失值个数为：{}".format(item, nulltotal))




# OutNominal()
# Num5()



#数据可视化
#直方图
def plothist(nums):
    plt.figure(figsize=(4,3))
    plt.hist(x=nums,  # 指定绘图数据
             bins=40,# 指定直方图中条块的个数
             edgecolor = 'black') # 指定直方图的边框色
    plt.ylabel('频数')
    plt.show()
# plothist(np.array(freq))

#盒图
def Box(nums):
    plt.figure(figsize=(4, 6))
    plt.boxplot(nums,notch=False)
    plt.ylabel('trending video numbers each day')
    plt.show()
# Box(np.array(freq))

def numBox():
    plt.figure(figsize=(30, 14), dpi=98)
    i=1
    for item in label_num:
        p=plt.subplot(3,2,i)
        plt.sca(p)
        pd.DataFrame(df[item]).boxplot()
        i=i+1
    plt.show()
# numBox()
def numhist():
    plt.figure(figsize=(30, 14), dpi=98)
    i=1
    for item in label_num:
        p=plt.subplot(3,2,i)
        plt.sca(p)
        df[item].hist(bins=50,edgecolor = 'black')
        p.set_title(item,fontsize=18)
        i=i+1
    plt.show()
# numhist()

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
    for i in range(len(nums['description'])):
        if (pd.isna(nums['description'][i])):
            nums.loc[i, 'description']  = nums.loc[i, 'title']
# corfill(df)
# n = [i for i in df['description'] if (pd.isna(i))]
# print("缺失值为：{}".format(len(n)))
# n=[i for i in df['tags'] if i=='[none]']
# len(n)
# corfill(df)
# n = [i for i in df['tags'] if i == '[none]']
# len(n)
n = [i for i in df['category_id'] if (pd.isna(i))]
print("缺失值为：{}".format(len(n)))



plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# for item in label_num:
#     lostdata(df, item)

# 查找类别名，若存在缺失，先用nan表示
# category = {}
# for i in j['items']:
#     category[i['id']] = i['snippet']['title']
# cate = []
# for i in df['category_id'].values:
#     if(str(i) in category.keys()):
#         cate.append(category[str(i)])
#     else:
#         cate.append('nan')
# freq = pd.Series(cate).value_counts()
# plothist(np.array(cate))
# Box(freq)
# plothist(freq)
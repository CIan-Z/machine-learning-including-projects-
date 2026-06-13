#贝叶斯：仅仅依赖概率就可以进行分类
#朴素：不考虑特征之间的关联性，特征之间必独立
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jieba  #分词包
from sklearn.feature_extraction.text import CountVectorizer#词频统计包
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB   # 多项分布朴素贝叶斯
import chardet#用来查看文件是什么类型编码

#1.读取文件
df = pd.read_csv('./data/书籍评价.csv',encoding='gbk')#显式设置编码为 gbk
#df.info()
#with open('./data/书籍评价.csv', 'rb') as f:  #自动检测 CSV 文件的字符编码
    #raw = f.read()  # 以二进制模式读取文件内容
    #result = chardet.detect(raw)  # 用 chardet 检测编码
    #print(result['encoding']) # 输出类似 'GB2312' 或 'utf-8' 的结果
#2.数据预处理
#2.1 添加labels列 充当：标签列。好评→1，差评→0   在文本分类任务中，将文本标签（如“好评”“差评”）映射为数字（1 和 0）是机器学习的硬性要求
df['labels'] = np.where(df['评价'] == '好评', 1, 0)#根据 df['好评'] 这一列的条件，生成一个由 1 和 0 组成的数组，然后赋值给 df['labels']。
                                                 #条件 df['好评'] == '好评' 为 True 的对应位置赋值为 1
#df['labels'] = df['labels'].map(lambda x: 1 if x==1 else 0)#遍历 df['labels'] 这一列的每一个值，如果值等于 1，就保留 1；否则全部变为 0
#df['labels'] = (df['评价'] == '好评').astype(int)#根据 '好评' 列是否为字符串 '好评'，生成二值标签 1 或 0，并赋值给 'labels' 列。
#df.info()
#print(df)
#2.2抽取labels列作为标签
y = df['labels']

#2.3 演示jieba分词
#print(jieba.lcut('好好学习，天天向上！'))

#2.4 对用户的评论信息，做切词。
#数据格式【【第一条评论，切词2，切词3。。。】，【第二条评论切词1，切词2，切词3.。。。】】no，不要这个
#comment_list = [jieba.lcut(line) for line in df['内容']]#如果没有拿到想要的，在df['内容']后面加个.values
#要这个['第一条切词1，切词2.。。。,'第二条切词1，切词2.。。。。]
comment_list = [','.join(jieba.lcut(line)) for line in df['内容']]
print(comment_list)
#演示字符串的join用法
#my_list = ['aa','bb','cc']
#print(','.join(my_list)
#2.5 加载 停用词列表：把那些不用参与模型预测的值删掉 例如：的，啊，哈。。。 停用词（Stop Words） 是在自然语言处理（NLP）中被设计为需要被过滤掉的常见词语。
with open('./data/stopwords.txt','r',encoding='utf-8') as src_f:
    stopwords_list = src_f.readlines()
    #算出每个字符后面的\n
    stopwords_list = [line.strip() for line in stopwords_list]
    #对停用表列词去重
    stopwords_list = list(set(stopwords_list))
    print(stopwords_list)
#2.6创建向量化对象，从评论切词列表中删除评论词，并统计词频
transfer = CountVectorizer(stop_words=stopwords_list)#依据停用词列表 将文本集合转换为词频矩阵（每篇文档中每个词语的出现次数）。会在构建词典时自动过滤掉 stopwords 列表中的词语
#2.7统计词频矩阵 先训练，后转换，再转数组
x = transfer.fit(comment_list)  #创建好的 transfer 是一个转换器，谁用它来 transform 文本，它就会把文本转换成词频数字矩阵
x = transfer.transform(comment_list).toarray()  #中文文本必须先分词并用空格分开（如用 jieba 分词），否则 CountVectorizer 会按空格切分，可能会把整句话当成一个词，导致完全错误。
print(x)
print(len(transfer.get_feature_names_out()))#删除后一共剩下的词的个数
#每个评论都会变成长度为37的列表→0没有这个词，1代表有这个词

#2.8因为就13条数据，因此把前10条当训练集，后3条当测试集
x_train = x[:10]
y_train = y[:10]
x_test = x[10:]
y_test = y[10:]

#3.特征工程
#4.模型训练
estimator = MultinomialNB()#创建朴素贝叶斯模型
estimator.fit(x_train, y_train)

#5. 模型预测
y_pred = estimator.predict(x_test)
print(f'预测结果：{y_pred}')

#6. 模型评估
print(f'准确率：{accuracy_score(y_test, y_pred)}')


















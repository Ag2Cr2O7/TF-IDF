##################################################################
#                     Python Homework                            #
#                       20221114249                              #
#                       XiangHengjing                            #
#                      计算机科学与技术                            #
#                        2022/10/27                              #
##################################################################
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from sklearn.datasets import fetch_20newsgroups
import pickle

#读取文本，存储当前文本的向量
class Sketch:
    def __init__(self,d,stop_name=None):
        #self.filename=txt_name #训练的txt文件
        self.d=d

    # 利用tf-idf转化为向量
    def tfidf_extractor(self, corpus):
        vectorizer = TfidfVectorizer(min_df=1,
                                     norm='l2',
                                     smooth_idf=True,
                                     stop_words='english',
                                     use_idf=True,
                                     ngram_range=(1,1),
                                     max_features=self.d)
        #序列作为参数传递给拟合器
        features = vectorizer.fit_transform(corpus)
        return vectorizer, features

    #读取文本文件
    def read_data(self,mode="train",filename=None):
        if filename: #从文档读取数据
            print('加载文件{}...'.format(filename))
            with open(filename, 'r',encoding='utf-8') as fp:
                labels = ""
                for label in tqdm(fp):
                    labels = labels + label
            return labels

        elif mode=="test": #拆分测试集
            print('加载fetch_20newsgroup--{0}数据集...'.format(mode))
            labels = fetch_20newsgroups(subset=mode).data
            mid=len(labels)//2
            l1,l2="",""
            for i in tqdm(range(mid)):
                l1+=labels[i]
            for i in tqdm(range(mid,len(labels))):
                l2+=labels[i]
            return l1,l2
        else:
            #读取训练数据集
            print('加载fetch_20newsgroup--{0}数据集...'.format(mode))
            labels = fetch_20newsgroups(subset=mode).data
            return labels

    def save1(self,p, filename="./train.model"): #保存模型
        f = open(filename, 'wb')
        pickle.dump(p, f)
        f.close()

    def load1(self,filename="./train.model"):
        f = open(filename, 'rb')
        p = pickle.load(f)
        f.close()
        for i in tqdm(range(len(p))):
            pass
        return p











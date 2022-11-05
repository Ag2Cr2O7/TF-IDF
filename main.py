from Sketch import *
from Vector import *
import nltk
##################################################################
#                     Python Homework                            #
#                       20221114249                              #
#                       XiangHengjing                            #
#                      计算机科学与技术                            #
#                        2022/10/27                              #
##################################################################
if __name__=="__main__":
    #for i in range(100,5000,1000): #消融实验解除该注释
        i=None
        # nltk.download('stopwords') #下载数据集
        # 实例化Sketch对象（向量的维度为d）
        MyVector = Sketch(d=i)  # 在文本文档中最多d个频繁的单词中创建一个特征矩阵
        # 读取文本训练文件
        # tokenized_corpus = MyVector.read_data(mode='train')
        # MyVector.save1(tokenized_corpus)
        print('加载训练数据...')
        tokenized_corpus = MyVector.load1(filename="./dataset/train.model")
        # 转化为tf-idf向量
        tfidf_vectorizer, tfidf_train_features = MyVector.tfidf_extractor(tokenized_corpus)

        print('加载测试文档...')
        text1 = MyVector.read_data(filename="./dataset/test1.txt")
        text2 = MyVector.read_data(filename="./dataset/test2.txt")
        text3 = MyVector.read_data(filename="./dataset/test3.txt")
        # 实例化计算相似度对象
        MyCompute = Vector()
        print('计算文本相似度...')
        dic = {"text1": [text1, text1], "text2": [text2, text3]}
        dist = MyCompute.tfidfvec_dis_list(tfidf_vectorizer, dic)
        print('文本1和文本2的相似度为：', dist[0])
        print('文本1和文本3的相似度为：', dist[1])
        # # 计算text1和text2向量的余弦相似度
        # dis1 = MyCompute.get_tfidfvec_dis(tfidf_vectorizer)
        # print('文本1和文本2的相似度为：', dis1)
        # # 计算text1和text3向量的余弦相似度
        # dis2 = MyCompute.get_tfidfvec_dis(tfidf_vectorizer)
        # print('文本1和文本3的相似度为：', dis2)







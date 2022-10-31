##################################################################
#                     Python Homework                            #
#                       20221114249                              #
#                          向恒京                                 #
#                      计算机科学与技术                             #
#                        2022/10/27                              #
##################################################################
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm
from nltk.corpus import stopwords

class Vector:
    def __init__(self):
        pass

    #对测试文档进行分词和去停词
    def del_stopwords(self, train_data):
        train_data = train_data.split()
        ans = [train_data[i] for i in tqdm(range(len(train_data))) if train_data[i] not in stopwords.words('english')]
        return str(ans)

    # 得到tfidf特征向量的余弦相似度
    def get_tfidfvec_dis(self,tfidf_vectorizer,text1,text2):
        fit_text1 = tfidf_vectorizer.transform([self.del_stopwords(text1)])
        fit_text2 = tfidf_vectorizer.transform([self.del_stopwords(text2)])
        vec1 = fit_text1.toarray()[0]
        vec2 = fit_text2.toarray()[0]
        return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))  # 使用cos相似度

    # 向量之间的距离（数组表示的实现）
    def tfidfvec_dis_list(self,tfidf_vectorizer,df):
        dis_list = []
        for text1, text2 in zip(df['text1'], df['text2']):
            dis_list.append(self.get_tfidfvec_dis(tfidf_vectorizer, text1, text2))
        return dis_list


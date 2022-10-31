# TF-IDF
This is the Python implementation for CCNU task

http://common.ai-augmented.com/user/login?cb=http%3A%2F%2Fxy.ai-augmented.com%2Fmycourse

>Xiang Hengjing 

## 任务描述

-通过计算并比较文档的摘要（或全文），实现文本的相似度比较。

-文档摘要的最简单形式可以使用文档中的k-grams（k个连续字符）的相对频率的向量来表示。把k-grams字符串s映射到0到d-1之间的整数，从而使得文档摘要向量的维度为d。

-通过sklearn的TfidfVectorizer 将文档摘要转化为一个向量维度为d的向量。

-创建文档摘要向量之后，可通过比较两个文档摘要向量的距离的方法来判断两个文档的相似度。

-实现Vector向量类进行文本向量转化和计算。

## Environment Requirement

The code runs well under python 3.7.13. The required packages are as follows:

- sklearn == 1.0.2
- nltk == 3.4
- numpy == 1.18.0
- tqdm == 4.50.2


## Quick Start
**Firstly**, Install the necessary libraries

**Secondly**, run [main.py](./main.py) in IDE or with command line:

```bash
python main.py
```

If the compilation is successful, the evaluator of python implementation will be called.


Model specific datasets are in configuration file *./dataset*.



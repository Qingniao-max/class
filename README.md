# 代码部署
<img src="images/1.png" width="800" alt="作业1">

# 优化特征选择方法
<img src="images/2.png" width="800" alt="作业2">

# 邮件分类项目

## 核心功能说明
本项目实现基于文本内容的邮件二分类（垃圾邮件/普通邮件），提供两种特征构建模式：
- 高频词特征选择模式
- TF-IDF特征加权模式

## 算法基础
### 多项式朴素贝叶斯分类器
采用贝叶斯定理进行概率估计，假设特征条件独立：
P(y|x_1,...,x_n) \propto P(y) \prod_{i=1}^{n}P(x_i|y) 

**具体应用形式**：
1. 计算先验概率： P(Spam) = \frac{垃圾邮件数}{总邮件数} 
2. 计算条件概率： P(单词_i|Spam) = \frac{单词_i在垃圾邮件中出现次数+1}{垃圾邮件总词数+唯一词数} 
3. 分类决策： argmax_{y} P(y) \prod_{i=1}^{n}P(x_i|y) 

## 数据处理流程
### 预处理步骤

1. 文件读取：读取UTF-8编码的文本文件
2. 无效字符过滤：正则表达式 [.【】0-9、——。，！~\*] 移除非文字字符
3. 中文分词：使用jieba进行精准模式分词
4. 停用词过滤：
   - 显式过滤：直接去除长度≤1的词语
   - 隐式过滤：通过后续特征选择实现
5. 文本标准化：转换为空格连接的分词字符串

## 特征构建流程
### 方法对比
<img src="images/3.png" width="500" alt="对比">

## 高频词/TF-IDF两种特征模式的切换方法
### 切换为高频词模式：
#### 在特征提取部分修改为：
from collections import Counter

def get_top_words(texts, top_n=100):
    all_words = chain(*[text.split() for text in texts])
    return [w for w,_ in Counter(all_words).most_common(top_n)]

top_words = get_top_words(train_texts)
vectorizer = CountVectorizer(vocabulary=top_words)

### 切换为TF-IDF模式：
#### 在特征提取部分修改为：
vectorizer = TfidfVectorizer(
    max_features=100,
    tokenizer=lambda x: x.split(),
    token_pattern=None
)



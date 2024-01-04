# -*- encoding:utf-8 -*-
import sys  # reload()之前必须要引入模块
reload(sys)
sys.setdefaultencoding('utf-8')

import time

import numpy as np
import theano
# theano.config.cxxflags = '-ID:\anaconda3\envs\cnn-bi-lstm-python27\share\mingwpy\include'
import cPickle
from collections import defaultdict
import re
import pandas as pd
import csv
from gensim.models import KeyedVectors
from tqdm import tqdm


# 加载数据，返回一个包含数据和词汇表信息的列表 revs, vocab
# datafile：包含数据的CSV文件路径
# cv：划分的折数
# clean_string：一个布尔值，表示是否要对文本进行清理，默认为True
def build_data_cv(datafile, cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    # 存储处理后的数据集,整理成一个字典
    revs = []
    # 记录每个单词的出现次数
    vocab = defaultdict(float)

    with open(datafile, "rb") as csvf:
        # delimiter = ','：指定CSV文件中字段之间的分隔符
        # quotechar = '"'：指定CSV文件中的引号字符
        csvreader = csv.reader(csvf, delimiter=',', quotechar='"')
        first_line = True
        for line in csvreader:
            # 跳过文件第一行的处理
            if first_line:
                first_line = False
                continue

            status = []
            # 使用正则表达式将每一行的第二个字段按照句号和问号进行分割，得到一个句子的列表
            # strip()方法是Python中的一个字符串方法，用于删除字符串左右两个的空格和特殊字符，如制表符、回车符、换行符等
            sentences = re.split(r'[.?]', line[1].strip())
            try:
                sentences.remove('')
            except ValueError:
                None

            for sent in sentences:
                if clean_string:
                    # 调用clean_str函数对句子进行清理
                    orig_rev = clean_str(sent.strip())

                    if orig_rev == '':
                        continue

                    # 将清理后的句子按空格分割成单词，去重得到单词集合
                    words = set(orig_rev.split())

                    # splitted是一个列表（List），存储了将清理后的句子orig_rev按空格分割后得到的单词序列
                    splitted = orig_rev.split()

                    # 如果句子长度超过150个单词，进行切分
                    if len(splitted) > 150:
                        orig_rev = []

                        # 计算切分的次数，每次切分20个单词
                        splits = int(np.floor(len(splitted) / 20))

                        # 将切分后的子句添加到orig_rev中
                        for index in range(splits):
                            orig_rev.append(' '.join(splitted[index * 20:(index + 1) * 20]))

                        # 处理剩余的单词，添加到orig_rev中
                        if len(splitted) > splits * 20:
                            orig_rev.append(' '.join(splitted[splits * 20:]))

                        status.extend(orig_rev)
                    else:
                        # 如果句子长度不超过150个单词，直接将清理后的句子添加到status中
                        status.append(orig_rev)
                else:
                    # 如果clean_string为假，表示不进行文本清理
                    # 将句子转换为小写并去除两端空白
                    orig_rev = sent.strip().lower()
                    words = set(orig_rev.split())
                    status.append(orig_rev)

                # 遍历单词集合，将每个单词添加到vocab字典中，并更新单词的出现次数
                for word in words:
                    vocab[word] += 1

            datum = {
                # 将 'y' 和其他值进行二元分类
                "y0": 1 if line[2].lower() == 'y' else 0,
                "y1": 1 if line[3].lower() == 'y' else 0,
                "y2": 1 if line[4].lower() == 'y' else 0,
                "y3": 1 if line[5].lower() == 'y' else 0,
                "y4": 1 if line[6].lower() == 'y' else 0,

                "text": status,
                "user": line[0],
                "num_words": np.max([len(sent.split()) for sent in status]),
                "split": np.random.randint(0, cv)}
            revs.append(datum)

    return revs, vocab

# 打开二进制文件 fname，加载词向量数据
# fname: Path to google word2vec file
def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}

    print '读取google word2vec file...'
    # 使用gensim加载Google预训练的Word2Vec模型
    model = KeyedVectors.load_word2vec_format(fname, binary=True)


    for word in tqdm(vocab, desc=r"加载词向量", unit="词", total=len(vocab)):
        # 检查单词是否在模型的词汇表中
        if word in model:
            word_vecs[word] = model[word]

    # print 'word_vecs',word_vecs
    return word_vecs

# 从预训练的词向量（word vectors）中构建一个词矩阵（word matrix），返回一个包含词矩阵和词索引映射的元组
def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    # 获取词向量字典中的词汇量大小
    vocab_size = len(word_vecs)
    # 创建一个空字典，用于将每个词映射到矩阵中的索引
    word_idx_map = dict()

    # 初始化一个全零矩阵，形状为(词汇量 + 1, k)，k是词向量的维度
    W = np.zeros(shape=(vocab_size + 1, k), dtype=theano.config.floatX)

    # 将第一行设为全零向量，对应未知词
    W[0] = np.zeros(k, dtype=theano.config.floatX)

    '''
    dtype = theano.config.floatX的作用是将NumPy数组的数据类型设置为与
    Theano默认浮点数类型相匹配的类型。这样做是为了确保NumPy数组的数据类型与后续在
    Theano中进行的运算一致，避免潜在的类型不匹配错误
    '''

    # 每个词的词向量填充到矩阵 W 的相应行，并将词映射到索引 i
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1

    # 每个词都可以通过索引在矩阵中找到对应的词向量
    return W, word_idx_map

# k 指定词向量的维度
def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        # 只有在训练数据中出现次数达到最小支持度的单词才被考虑
        if word not in word_vecs and vocab[word] >= min_df:
            # 为当前单词生成一个随机的词向量，生成一个在指定范围内均匀分布的随机数
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)
            # 显示在训练数据中出现，但在预训练词向量中未见的单词
            # print (word)


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    # 保留字母、数字、逗号、感叹号、问号、单引号和反引号，移除其他字符
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)

    # 将缩写形式转换为全写形式
    string = re.sub(r"\'s", " \'s ", string)
    string = re.sub(r"\'ve", " have ", string)
    string = re.sub(r"n\'t", " not ", string)
    string = re.sub(r"\'re", " are ", string)
    string = re.sub(r"\'d", " would ", string)
    string = re.sub(r"\'ll", " will ", string)

    #  在逗号、感叹号、左括号、右括号、问号前后添加空格
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " \? ", string)

    #    string = re.sub(r"[a-zA-Z]{4,}", "", string)
    string = re.sub(r"\s{2,}", " ", string)

    # 当 TREC 为 True 时，保留字符的大小写
    return string.strip() if TREC else string.strip().lower()


def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    #  保留字母、数字、逗号、感叹号、问号、单引号和反引号，移除其他字符
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    # 移除多余的空格，将两个或更多连续的空格替换为一个空格
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


#  返回 Mairesse 特征的字典，AUTHID 作为外键
def get_mairesse_features(file_name):
    feats = {}
    with open(file_name, "rb") as csvf:
        csvreader = csv.reader(csvf, delimiter=',', quotechar='"')
        for line in csvreader:
            # 将每行的第一个元素（AUTHID）作为键，将其余部分转换为浮点数并作为值
            feats[line[0]] = [float(f) for f in line[1:]]
    return feats


if __name__ == "__main__":
    start_time = time.time()
    # Path to google word2vec file
    w2v_file = r"C:\Users\LeviDIAO\Desktop\Dataset\GoogleNews-vectors-negative300.bin"
    # Path to essays.csv file containing the annotated dataset
    data_folder = r"C:\Users\LeviDIAO\Desktop\Dataset\essays2.csv"
    # Path to mairesse.csv containing Mairesse features for each sample / essay
    mairesse_file = r"C:\Users\LeviDIAO\Desktop\Dataset\mairesse.csv"

    # 加载文本数据
    print ("loading data...")
    # 10折交叉验证
    revs, vocab = build_data_cv(data_folder, cv=10, clean_string=True)
    # 提取每个样本的词汇数量
    num_words = pd.DataFrame(revs)["num_words"]
    # 计算样本中最大的词汇数量
    max_l = np.max(num_words)
    print ("data loaded!")
    print ("number of status: " + str(len(revs)))
    print ("vocab size: " + str(len(vocab)))
    print ("max sentence length: " + str(max_l))
    end_time = time.time()
    elapsed_time = end_time - start_time
    # 打印消耗的时间
    print("代码运行时间: %s 秒" % elapsed_time)

    # 加载Word2Vec向量
    print ("loading word2vec vectors...")
    w2v = load_bin_vec(w2v_file, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    end_time = time.time()
    elapsed_time = end_time - start_time
    # 打印消耗的时间
    print("代码运行时间: %s 秒" % elapsed_time)

    # 将在训练数据中出现但不在Word2Vec模型中的单词添加到模型中，使用的是预训练的 Word2Vec 模型
    add_unknown_words(w2v, vocab)
    # 获取用于模型训练的Word2Vec向量和单词索引映射，用作初始词向量
    W, word_idx_map = get_W(w2v)


    rand_vecs = {}
    # 将训练数据中出现但不在 Word2Vec 模型中的单词添加到 rand_vecs 中，使用的是随机生成的词向量
    add_unknown_words(rand_vecs, vocab)
    # 用作那些未在预训练模型中找到的单词的初始词向量
    W2, _ = get_W(rand_vecs)

    # 加载Mairesse特征
    mairesse = get_mairesse_features(mairesse_file)

    cPickle.dump([revs, W, W2, word_idx_map, vocab, mairesse], open("essays_mairesse_small_batch.p", "wb"))
    print "dataset created!"
    end_time = time.time()
    elapsed_time = end_time - start_time
    # 打印消耗的时间
    print("代码运行时间: %s 秒" % elapsed_time)

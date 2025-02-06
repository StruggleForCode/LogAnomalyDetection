"""
Contributor:  Micheleen Harris
Date:  Feb. 20, 2022
Original source:  https://github.com/donglee-afar/logdeep/issues/3#issuecomment-750028771
Purpose:  Map event ids to an encoded semantics vector (specifically for loganomaly method)
         将事件 ID 映射到一个编码后的语义向量（专门用于 LogAnomaly 方法）。
Notes:
- Uses the spellpy parser project:  https://github.com/nailo2c/spellpy (need to pip install)
- Need the stop words Python file from SpaCy in project folder with this file:  https://github.com/explosion/spaCy/blob/master/spacy/lang/en/stop_words.py
- Example below are from Ubuntu system logs (normal and abnormal as deemed by user)

	•	本代码使用 spellpy 解析器项目：spellpy（需要使用 pip install 安装）。
	•	需要 SpaCy 的 停用词列表，请将以下文件放入项目文件夹中：stop_words.py。
	•	下面的示例数据来自 Ubuntu 系统日志（包括正常日志和用户认为的异常日志）。

Get "cc.en.300.vec" by (on Linux; note, the unarchived file is ~4.5 GB):
获取 FastText 预训练词向量 cc.en.300.vec（Linux 命令，解压后约 4.5GB）：

mkdir vec_models
cd vec_models
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz
gunzip cc.en.300.vec.gz

源码链接：
https://gist.github.com/michhar/388d037439da6114d67aa8f793293870
"""
import re
import json
import io
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from collections import OrderedDict, Counter
import math
from pprint import pprint
from datetime import datetime

from stop_words import StopWords

from spellpy import spell


class Preprocessor:

    def __init__(self):
        self.stop_words = StopWords().STOP_WORDS

    def df_transfer(self, df, event_id_map):
        # 通过将年份字符串重复 df.shape[0] 次，创建一个包含相同年份字符串的列表，列表的长度等于 df 的行数。
        year = [str(datetime.utcnow().year)] * df.shape[0]
        timestamps = list(map(lambda a, b, c, d: a + '-' + b + '-' + str(c).rstrip() + ' ' + str(d),
                              year,
                              df['Month'],
                              df['Day'],
                              df['Time']))
        df['datetime'] = pd.to_datetime(timestamps, format='%Y-%b-%d.0 %H:%M:%S', errors='coerce')
        df.dropna(inplace=True)
        df = df[['datetime', 'EventId']]
        # df['EventId'] = df['EventId'].apply(lambda e: event_id_map[e] if event_id_map.get(e) else -1)
        # df['EventId'] = df['EventId'].apply(lambda e: event_id_map.get(e, -1))
        # 将EventId列的值映射为新的值，如果没有映射到值则使用-1
        df.loc[:, 'EventId'] = df['EventId'].apply(lambda e: event_id_map.get(e, -1))
        # 按分钟重新采样日志数据，并应用自定义的采样方法
        deeplog_df = df.set_index('datetime').resample('1min').apply(self._custom_resampler).reset_index()
        return deeplog_df

    def _custom_resampler(self, array_like):
        """Can sample however is needed"""
        return list(array_like)

    def file_generator(self, filename, df):
        # 确保目录存在
        dir_name = os.path.dirname(filename)  # 获取目录路径
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)  # 递归创建目录

        # 确保文件存在
        if not os.path.exists(filename):
            open(filename, 'w').close()  # 创建空文件

        with open(filename, 'w') as f:
            for event_id_list in df['EventId']:
                for event_id in event_id_list:
                    f.write(str(event_id) + ' ')
                if len(event_id_list) > 0:  # 过滤空行
                    f.write('\n')

    # 该方法的作用是对文本进行标准化（Normalization），
    # 去除特殊字符、转换大小写格式、处理驼峰命名（Camel Case）、去除停用词，并将其转换为规范的单词列表（tokens），适用于自然语言处理（NLP）任务。
    def normalize_text(self, text):
        """
        Normalize text to extract most salient tokens
        Ref: https://github.com/MLWorkshops/nlp-dealing-with-text-data/blob/master/Dealing-with-text-data.ipynb
        Ref: turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
        """
        # replace special characters with space and remove digits
        # \W+：匹配所有非字母、非数字、非下划线的字符（即特殊字符和标点），并替换为空格 ' '
        text = re.sub(r'\W+', ' ', text)
        # \d：匹配所有数字，并将其替换为空字符串 ''，即删除所有数字。
        text = re.sub('\d', '', text)
        # convert camel case to snake case, then replace _ with space
        # 这两行代码的作用是将驼峰命名（CamelCase）转换为小写、带空格的格式，以便后续的分词处理。
        # text = "SystemFailure detected on ServerNode3"
        # # 处理步骤：
        # # "SystemFailure" -> "System_Failure"
        # # "ServerNode3" -> "Server_Node3"
        # # 转换为小写并去掉下划线：
        # text = "system failure detected on server node3"
        text = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
        text = re.sub('([a-z0-9])([A-Z])', r'\1_\2', text).lower().replace('_', ' ')
        # tokenize, removing stop words (from SpaCy)
        # 去除停用词
        normalized_tokens = [t for t in text.split(' ') if t not in self.stop_words and t != '']
        return normalized_tokens


    def dump2json(self, dump_dict, target_path):
        """
        Save json and any bytes-like objects to file
        该方法的作用是将 字典对象 dump_dict 保存为 JSON 格式的文件，并处理任何 bytes 类型的对象。
        该方法通过自定义 JSONEncoder 类来解决字节类型 (bytes) 对象的序列化问题。
        它将所有内容以 JSON 格式保存到指定的 target_path 文件中。
        """

        class MyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, bytes):
                    return str(obj, encoding='utf-8')
                return json.JSONEncoder.default(self, obj)

        with open(target_path, 'w', encoding='utf-8') as file:
            file.write(json.dumps(dump_dict, cls=MyEncoder, indent=4))

    '''
        1.	create_word2idf 方法的目的是通过日志数据计算每个单词的 IDF（逆文档频率） 值。
        2.	idf_matrix 存储了每个文档中每个单词的出现次数。
        3.	df_vec 计算每个单词的文档频率（DF）。
        4.	idf_vec 使用平滑的公式计算了每个单词的 IDF 值。
        5.	通过将 idf_vec 应用到词频矩阵 X 中，生成了包含 IDF 权重的新矩阵 idf_matrix。
        6.	word2idf 字典存储了每个单词及其 IDF 值，并为未登录词（OOV）提供了平滑处理。
    
        最终返回一个 单词到 IDF 值的映射，用于 TF-IDF 计算，常用于文本分析和特征提取。
    '''
    def create_word2idf(self, log_train, eventid2template):
        """
        Create a word to IDF dict
        TF = term frequency
        IDF = inverse document frequency
        这段代码的作用是通过训练日志数据来计算 词的逆文档频率（IDF），并返回一个 word2idf 字典，
        其中包含了每个词的 IDF 值。它基于 TF-IDF（词频-逆文档频率）模型的计算方法，其中 TF 是词频，IDF 是逆文档频率。
        """
        # idf_matrix 是一个列表，包含所有事件对应的单词（通过 eventid2template 映射得到）。
        # log_train['EventId'] 是日志数据中的事件 ID 列，其中每个 seq 是一个事件 ID 列表。
        # 将每个事件 ID 映射到其对应的单词列表，最终得到 idf_matrix，它是一个包含所有事件模板单词的矩阵。
        idf_matrix = list()
        for seq in log_train['EventId']:
            for event in seq:
                mapped_eid = eventid2template.get(event, -1)
                if mapped_eid != -1 and mapped_eid:
                    idf_matrix.append(eventid2template[event])
                else:
                    print(f"Warning: Event ID {eid} not found in eventid2template in idf")

        # 确定最长的子列表长度
        max_len = max(len(sublist) for sublist in idf_matrix)

        # 填充每个子列表，使其长度一致
        idf_matrix = [sublist + [0] * (max_len - len(sublist)) for sublist in idf_matrix]

        # 转换为 NumPy 数组
        idf_matrix = np.array(idf_matrix)

        # Counter(idf_matrix[i])：对于 idf_matrix 中的每一行（即每个事件的单词列表），统计每个单词的出现次数。
        # word_counts 是一个 Counter 对象，它返回每个单词在该事件中的出现频次。
        X_counts = []
        for i in range(idf_matrix.shape[0]):
            word_counts = Counter(idf_matrix[i])
            X_counts.append(word_counts)

        # 将 X_counts 列表转换为一个 DataFrame，每一行代表一个事件，每一列代表一个单词，值表示该单词在该事件中的出现次数。
        # fillna(0)：如果某个事件中没有某个单词，则填充为 0。
        X_df = pd.DataFrame(X_counts)
        X_df = X_df.fillna(0)

        # X_df.columns：获取所有单词的列名，存储在 events 中。
        # X_df.values：获取 DataFrame 的数值部分，存储为矩阵 X。
        # num_instance, num_event = X.shape：获取矩阵的形状，num_instance 是文档的数量（事件的数量），num_event 是词汇表中单词的数量。
        # df_vec = np.sum(X > 0, axis=0)：计算每个单词在多少文档中出现（即文档频率 DF）。
        # X > 0 生成一个布尔矩阵，np.sum(axis=0) 对列求和，得到每个单词的文档频率。
        events = X_df.columns
        X = X_df.values
        num_instance, num_event = X.shape
        df_vec = np.sum(X > 0, axis=0)

        # smooth idf like sklearn
        # IDF 计算公式：
        # num_instance 是总文档数（即事件数）。
        # df_vec 是每个单词出现的文档数（即文档频率）。
        # np.log((num_instance + 1) / (df_vec + 1)) + 1 是带有平滑的 IDF 计算公式，避免出现 log(0) 的情况，平滑因子是 +1。
        idf_vec = np.log((num_instance + 1) / (df_vec + 1)) + 1
        print(idf_vec)

        # np.tile(idf_vec, (num_instance, 1))：
        #   将 idf_vec 向量复制 num_instance 次，生成一个矩阵，每一列都包含该单词的 IDF 值。
        # X * np.tile(idf_vec, (num_instance, 1))：
        #   将 IDF 向量应用到 X 矩阵中，即每个单词的词频乘以该单词的 IDF 值，得到 IDF 权重的矩阵。
        idf_matrix = X * np.tile(idf_vec, (num_instance, 1))
        X_new = idf_matrix

        # zip(events, idf_vec)：将所有单词与其对应的 IDF 值配对。
        # word2idf[i] = j：将每个单词及其 IDF 值存入 word2idf 字典。
        # word2idf['oov']：为 OOV（Out-Of-Vocabulary）词汇 设置一个默认的 IDF 值。
        # 这里的 math.log((num_instance + 1) / (29 + 1)) + 1 是一种处理未在词汇表中的词汇的平滑方法。
        word2idf = dict()
        for i, j in zip(events, idf_vec):
            word2idf[i] = j
            # smooth idf when oov
            word2idf['oov'] = (math.log((num_instance + 1) / (29 + 1)) + 1)
        return word2idf

    def create_semantic_vec(self, eventid2template, fasttext_map, word2idf):
        event2semantic_vec = dict()
        for event in eventid2template.keys():
            template = eventid2template[event]
            tem_len = len(template)
            count = dict(Counter(template))
            for word in count.keys():
                # TF
                TF = count[word] / tem_len
                # IDF
                IDF = word2idf.get(word, word2idf['oov'])
                count[word] = TF * IDF
            value_sum = sum(count.values())
            for word in count.keys():
                count[word] = count[word] / value_sum
            semantic_vec = np.zeros(300)
            for word in count.keys():
                try:
                    fasttext_weight = np.array(fasttext_map[word])
                except KeyError as ke:
                    # word not in fasttext
                    pass
                semantic_vec += count[word] * fasttext_weight
            event2semantic_vec[event] = list(semantic_vec)
        return event2semantic_vec


class FastTextProcessor:
    """Use fasttext vectors to generate map"""

    def __init__(self):
        self.template_set = set() # 存储日志模板中的所有唯一单词。
        self.template_fasttext_map = {} # 存储每个单词对应的 FastText 词向量，形成单词到向量的映射。

    # 所有单词去重存储，供后续 FastText 处理。
    def create_template_set(self, result):
        print('Creating template set')
        for key in tqdm(result.keys()):
            for word in result[key]:
                self.template_set.add(word)

    def load_vectors(self, fname):
        """
        Ref: https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md
        """
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}
        print('Loading vectors')
        for line in tqdm(fin):
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = map(float, tokens[1:])
        return data

    def create_map(self):
        fasttext = self.load_vectors(os.path.join('vec_models', 'wiki-news-300d-1M.vec'))
        print('Creating fasttext map')
        for word in tqdm(self.template_set):
            try:
                self.template_fasttext_map[word] = list(fasttext[word])
            except KeyError as ke:
                # fasttext does not have word
                pass
        return self.template_fasttext_map


if __name__ == "__main__":

    preprocessor = Preprocessor()

    ##########
    # Parser #
    ##########
    input_dir = '../data/hdfs/'
    output_dir = './results_spell/'
    recreated_parse_logs = True
    # "Content" is like the log message - what we want to parse
    # the following is specific to the syslog, so match to those "columns"
    # <Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>
    # log_format = '<Month> <Day> <Time> <MachineName> <Content>'
    #log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'
    log_format = '<Month> <Day> <Time> <Level> <Component>(\[<PID>\])?: <Content>'
    log_main = 'Linux'
    tau = 0.55

    parser = spell.LogParser(
        indir=input_dir,
        outdir=output_dir,
        log_format=log_format,
        logmain=log_main,
        tau=tau,
    )

    # if the we wish, we can recreate the parsed csv's
    if recreated_parse_logs:
        os.makedirs(output_dir)
        for log_name in ['Linux_2k.log']:
            parser.parse(log_name)

    ##################
    # Transformation #
    ##################
    # TODO:  read from object, not file i/o again
    df_train = pd.read_csv(f'{output_dir}/Linux_2k.log_structured.csv')
    # df_test_normal = pd.read_csv(f'{output_dir}/syslog.2.updated_structured.csv')
    # df_test_abnormal = pd.read_csv(f'{output_dir}/abnormal_states.log_structured.csv')

    print('Number of classes for training = ', df_train['EventId'].unique().shape)

    event_id_map = dict()
    for i, event_id in enumerate(df_train['EventId'].unique(), 1):
        event_id_map[event_id] = i

    # Train Set
    log_train = preprocessor.df_transfer(df_train, event_id_map)
    preprocessor.file_generator('./results_preprocessor/train', log_train)

    # Test Normal Set
    # log_test_normal = preprocessor.df_transfer(df_test_normal, event_id_map)
    # preprocessor.file_generator('./results_preprocessor/test_normal', log_test_normal)

    # Test Abnormal Set
    # log_test_abnormal = preprocessor.df_transfer(df_test_abnormal, event_id_map)
    # preprocessor.file_generator('./results_preprocessor/test_abnormal', log_test_abnormal)

    #####################
    # Event to Template #
    #####################
    eventid2template = {}
    print('Creating event IDs to templates')
    for eid in tqdm(df_train['EventId'].unique()):
        # 通过 get 获取 event_id_map 中的映射值，如果不存在，返回 -1
        mapped_eid = event_id_map.get(eid, -1)

        # 如果映射值不是 -1 且在 df_train 中存在该行，才进行处理
        if mapped_eid != -1 and mapped_eid in df_train.index:
            eventid2template[mapped_eid] = preprocessor.normalize_text(
                df_train.loc[mapped_eid, 'EventTemplate'])
        else:
            print(f"Warning: Event ID {eid} not found in event_id_map or df_train.")
            # eventid2template[mapped_eid] = []  # 或者为其分配一个空值，视你的需求而定
    preprocessor.dump2json(eventid2template, './results_preprocessor/eventid2template.json')

    ################
    # Fasttext map #
    ################
    fasttext_processor = FastTextProcessor()
    fasttext_processor.create_template_set(eventid2template)
    template_fasttext_map = fasttext_processor.create_map()
    preprocessor.dump2json(template_fasttext_map, './results_preprocessor/fasttext_map.json')

    ###############
    # Word to IDF #
    ###############
    word2idf = preprocessor.create_word2idf(log_train, eventid2template)
    preprocessor.dump2json(word2idf, './results_preprocessor/word2idf.json')

    #############################
    # Event to Semantics Vector #
    #############################
    event2semantic_vec = preprocessor.create_semantic_vec(eventid2template, template_fasttext_map, word2idf)
    preprocessor.dump2json(event2semantic_vec, './results_preprocessor/event2semantic_vec.json')

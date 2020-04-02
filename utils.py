# encoding: utf-8

import os
import re
import jieba
import numpy as np
import pickle as pkl
import gensim.models.word2vec as word2vec

from collections import defaultdict

people_name_list = [
    '沙瑞金', '田国富', '高育良', '侯亮平', '钟小艾', '陈岩石', '欧阳菁', '易学习', '王大路', '蔡成功', '孙连城', '季昌明', '丁义珍',
    '郑西坡', '赵东来', '高小琴', '赵瑞龙', '林华华', '陆亦可', '刘新建', '刘庆祝'
]
for name in people_name_list:
    jieba.suggest_freq(name, True)

WORD_START = "WORD_START"
WORD_END = "WORD_END"

stop_words = {"都", "就", "还", "有", "和", "吧", "啥", "又", "的", "了", "在", "是", "把", "中", "像", "却", "做", "能", "会",
              "嘛", "那", "为", "老", "才", "啊", "用", "找", "再", "厂", "大", "事", "个", "时", "里", "谁", "外", "哎", "“",
              "“", "”", "与", "我", "顶", "然后", "他", "它", "前", "这", "上", "呀", "也", "啦", "但", "不", "出", "着", "并",
              "紧", "细", "…", "你", "说", "让", "叫", "多", "少", "年", "地", "新", "过来", "这时", "如果", "一个", "真", "后",
              "可能", "看着", "易", "一个", "我们", "从", "一般", "一", "开始", "出去", "真有", "他", "去", "下", "很", "给", "人",
              "这个", "吗", "很", "可", "看", "之间", "—", "向", "过", "还是", "走", "干吗", "什么", "虽", "旁", "起来", "你们",
              "因", "道", "便", "听", "送", "搞"}


def process_wiki():
    count_dict, all_word_list = defaultdict(int), []
    for sen in word2vec.Text8Corpus('data/enwik8'):
        for word in sen:
            word = re.sub(r"[^a-z.]", "", word)
            if len(word) == 0:
                continue
            print(word)
            input()
            count_dict[word] += 1
            all_word_list.append(word)
    word_list = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)[150:]
    word_list = map(lambda x: x[0], filter(lambda x: x[1] > 10, word_list))

    word_ind_dict = {}
    for ind, word in enumerate(word_list):
        word_ind_dict[word] = ind
    word_ind_dict[WORD_START] = len(word_ind_dict)
    word_ind_dict[WORD_END] = len(word_ind_dict)

    def filter_func(x):
        if x == ".":
            return False
        return x not in word_ind_dict

    all_sentences = " ".join(list(filter(filter_func, all_word_list))).split(".")
    print(all_sentences[:100])


def process_raw():
    with open("data/tweet_global_warming.txt") as fd:
        for ind, line in enumerate(fd):
            line = line.strip().split(",")
            new_line = " ".join(line[:-2])
            new_line = re.sub(r"[^a-zA-Z ]|link", "", new_line)
            new_line = re.sub(r" +", " ", new_line)

            yield new_line


def get_ind(word_ind_dict, word_list, ind):
    if ind < 0:
        word = WORD_START
    elif ind >= len(word_list):
        word = WORD_END
    else:
        word = word_list[ind]
    return word_ind_dict.get(word, None)


def generate_sample_list(min_count=4, window_size=4):
    ind_file_name = "models/skip_gram/word_ind.pkl"
    samples_file_name = "models/skip_gram/samples.pkl"

    if os.path.exists(ind_file_name) and os.path.exists(samples_file_name):
        with open(ind_file_name, "rb") as fd:
            word_ind_dict = pkl.load(fd)
        with open(samples_file_name, "rb") as fd:
            sample_list = pkl.load(fd)
        return word_ind_dict, sample_list

    count_dict = defaultdict(int)
    for sen in process_raw():
        for word in sen.strip().split(" "):
            count_dict[word] += 1

    word_ind_dict, start_ind = {WORD_START: 0, WORD_END: 1}, 2
    for word, count in count_dict.items():
        if count <= min_count or word in word_ind_dict:
            continue
        word_ind_dict[word] = start_ind
        start_ind += 1

    sample_list = []
    for sen in process_raw():
        word_list = sen.strip().split(" ")
        for center_ind in range(len(word_list)):
            for shift in range(1, window_size+1):
                center = get_ind(word_ind_dict, word_list, center_ind)
                if center is None:
                    continue
                left = get_ind(word_ind_dict, word_list, center_ind - shift)
                right = get_ind(word_ind_dict, word_list, center_ind + shift)
                for label in [left, right]:
                    if label is None:
                        continue
                    sample_list.append([center, label])

    with open(ind_file_name, "wb") as fd:
        pkl.dump(word_ind_dict, fd)
    with open(samples_file_name, "wb") as fd:
        pkl.dump(sample_list, fd)

    return word_ind_dict, sample_list


def process_chinese(min_count=10):
    dir_name = "data/books"
    count_dict, sentences_list = defaultdict(int), []
    for file_name in os.listdir(dir_name):
        with open(os.path.join(dir_name, file_name)) as fd:
            txt = fd.read().strip()
            txt = re.sub(r"\s+| +|\n+", "", txt)

        line_list = re.split(r"，|。|？|：|！", txt)
        for line in line_list:
            word_list = list(jieba.cut(line))
            for word in word_list:
                count_dict[word] += 1
            sentences_list.append(word_list)
    word_list = filter(lambda x: x[1] >= min_count, sorted(count_dict.items(), key=lambda x: x[1], reverse=True))
    word_ind_dict, ind = {}, 0
    for word, count in word_list:
        word = word.strip()
        if word in stop_words:
            continue
        word_ind_dict[word] = ind
        ind += 1

    return count_dict, word_ind_dict, sentences_list


def get_ind_chinese(word_ind_dict, word_list, ind):
    if ind < 0 or ind >= len(word_list):
        return None
    word = word_list[ind].strip()
    return word_ind_dict.get(word, None)


def generate_sample_list_chinese(min_count=10, window_size=4, cache=False):
    ind_file_name = "models/skip_gram/word_ind.pkl"
    samples_file_name = "models/skip_gram/samples.pkl"

    if cache:
        if os.path.exists(ind_file_name) and os.path.exists(samples_file_name):
            with open(ind_file_name, "rb") as fd:
                word_ind_dict = pkl.load(fd)
            with open(samples_file_name, "rb") as fd:
                sample_list = pkl.load(fd)
            return word_ind_dict, sample_list

    sample_list = []
    count_dict, word_ind_dict, sentences_list = process_chinese(min_count=min_count)
    for word_list in sentences_list:
        word_list = list(filter(lambda x: x in word_ind_dict, word_list))
        for center_ind in range(len(word_list)):
            for shift in range(1, window_size+1):
                center = get_ind_chinese(word_ind_dict, word_list, center_ind)
                if center is None:
                    break
                left = get_ind_chinese(word_ind_dict, word_list, center_ind - shift)
                right = get_ind_chinese(word_ind_dict, word_list, center_ind + shift)
                for label in [left, right]:
                    if label is not None:
                        sample_list.append([center, label])

    if cache:
        with open(ind_file_name, "wb") as fd:
            pkl.dump(word_ind_dict, fd)
        with open(samples_file_name, "wb") as fd:
            pkl.dump(sample_list, fd)

    count_list = list(count_dict.values())
    print("word 25%: {0}, 50%: {1}, 75%: {2}, 100%: {3}".format(
        np.percentile(count_list, 25),
        np.percentile(count_list, 50),
        np.percentile(count_list, 75),
        np.percentile(count_list, 100)
    ))
    print("ind dict num: {0}".format(len(word_ind_dict)))
    print("sentences num: {0}".format(len(sentences_list)))
    print("samples num: {0}".format(len(sample_list)))

    return word_ind_dict, sample_list


if __name__ == "__main__":
    ind_dict, samples = generate_sample_list_chinese()


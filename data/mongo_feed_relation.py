# -*- coding: utf-8 -*-
from pymongo import MongoClient
import unicodedata
import pandas as pd
import statistics

################################# for tokenize ##########################################
def is_punctuation(ch):
    code = ord(ch)
    return 33 <= code <= 47 or \
           58 <= code <= 64 or \
           91 <= code <= 96 or \
           123 <= code <= 126 or \
           unicodedata.category(ch).startswith('P')

def is_cjk_character(ch):
    code = ord(ch)
    return 0x4E00 <= code <= 0x9FFF or \
           0x3400 <= code <= 0x4DBF or \
           0x20000 <= code <= 0x2A6DF or \
           0x2A700 <= code <= 0x2B73F or \
           0x2B740 <= code <= 0x2B81F or \
           0x2B820 <= code <= 0x2CEAF or \
           0xF900 <= code <= 0xFAFF or \
           0x2F800 <= code <= 0x2FA1F

def is_space(ch: str) -> bool:
    return ch == ' ' or ch == '\n' or ch == '\r' or ch == '\t' or unicodedata.category(ch) == 'Zs'

def is_control(ch):
    return unicodedata.category(ch) in ('Cc', 'Cf')

def tokenize(text):
    text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
    text = text.lower()
    spaced = ''
    for ch in text:
        if is_punctuation(ch) or is_cjk_character(ch) or ch.isdigit():
            spaced += ' ' + ch + ' '
        elif is_space(ch):
            spaced += ' '
        elif ord(ch) == 0 or ord(ch) == 0xfffd or is_control(ch):
            continue
        else:
            spaced += ch

    return spaced.strip().split()

################################################################################

def feed_mongo(relation_filename, db_name, collection_name):
    with open(relation_filename, "r", encoding="utf-8") as f:
        content = [_.strip() for _ in f.readlines()]
        data_set = []
        for line in content:
            parts = line.split()
            label, text = parts[0], ''.join(parts[1:])

            if label != "0":
                data_set.append({
                    "x": tokenize(text),
                    "y": label
                })

    if collection_name == "train":
        df_data = pd.DataFrame(data_set)
        df_label_count = df_data[['y']].groupby(["y"]).size().reset_index(name="count")
        median_len = int(statistics.median(list(df_label_count['count'].values)))

        df_all = pd.DataFrame(columns=["x", "y"])
        for index, content in df_data.groupby(["y"]):
            df_all = pd.concat([df_all, content.sample(n=median_len, replace=True)], ignore_index=True)

        data_set = df_all.to_dict("records")

    client = MongoClient("mongodb://localhost:27017")
    db = client[db_name]
    collection = db[collection_name]

    batch_size = 500
    batch = (len(data_set) // batch_size) + 1

    for i in range(batch):
        part_ds = data_set[batch_size * i: batch_size * (i + 1)]
        if part_ds:
            collection.insert_many(part_ds)

### train
feed_mongo("train_relation.txt", "classification", "train")

### test
feed_mongo("test_relation.txt", "classification", "test")
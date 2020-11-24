# -*- coding: utf-8 -*-
import json
import unicodedata

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
# 加工成BIO标注序列
def bio_sent(sent, spo_list):
    bio_list = ['O'] * len(sent)
    for item in spo_list:
        subj = item[0]
        len_subj = len(tokenize(subj))
        obj = item[-1]
        len_obj = len(tokenize(obj))

        for i in range(0, len(sent) - len_subj + 1):
            if "".join(sent[i:i + len_subj]) == subj:
                bio_list[i] = 'B-SUBJ'
                for j in range(1, len_subj):
                    bio_list[i + j] = 'I-SUBJ'

        for i in range(0, len(sent) - len_obj + 1):
            if "".join(sent[i:i + len_obj]) == obj:
                bio_list[i] = 'B-OBJ'
                for j in range(1, len_obj):
                    bio_list[i + j] = 'I-OBJ'
    return sent, bio_list

# 加载数据集
def load_data(input_filename, output_filename):
    output_str = ""
    with open(input_filename, 'r', encoding='utf-8') as f:
        content = f.readlines()

    for l in content:
        l = json.loads(l)
        sentence = tokenize(l['text'])
        spo_list = [(spo['subject'], spo['predicate'], spo['object']) for spo in l['spo_list']]
        sent, bio_sentence = bio_sent(sentence, spo_list)
        for char, tag in zip(sent, bio_sentence):
            output_str += char + " " + tag + "\n"

        output_str += "\n"

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(output_str)

################################## main function ##########################################
train_input = "train_data.json"
train_output = "train_bio.txt"
load_data(train_input, train_output)

test_input = "test_data.json"
test_output = "test_bio.txt"
load_data(test_input, test_output)


from pymongo import MongoClient

def split_word_tag(wt_list):
    """
    Input: ["a O", "b O", "c B-SUBJ", "d I-SUBJ", "e O"]
    Output: [["a", "b", "c", "d", "e"], ["O"， “O”, "B-SUBJ", "I-SUBJ", "O"]]
    """
    wt_list = [wt.split(" ") for wt in wt_list]
    word_tuple, tag_tuple = list(zip(*wt_list))
    return [list(word_tuple), list(tag_tuple)]

with open("test_bio.txt", "r", encoding="utf-8") as f:
    content = f.read()
    sentences = list(filter(lambda x: x, content.split("\n\n")))
    sentences = [split_word_tag(sen.split("\n")) for sen in sentences]
    train_x, train_y = list(zip(*sentences))

    batch_size = 500
    batch = (len(train_x) // batch_size) + 1

    client = MongoClient("mongodb://localhost:27017")
    db = client["spo"]
    collection = db["train"]
    for i in range(batch):
        part_x = train_x[batch_size * i: batch_size * (i + 1)]
        part_y = train_y[batch_size * i: batch_size * (i + 1)]

        record_list = []
        for j in range(batch_size):
            try:
                record_list.append({"x": part_x[j], "y": part_y[j]})
            except:
                break

        if record_list:
            collection.insert_many(record_list)


with open("test_bio.txt", "r", encoding="utf-8") as f:
    content = f.read()
    sentences = list(filter(lambda x: x, content.split("\n\n")))
    sentences = [split_word_tag(sen.split("\n")) for sen in sentences]
    test_x, test_y = list(zip(*sentences))

    batch_size = 500
    batch = (len(train_x) // batch_size) + 1

    client = MongoClient("mongodb://localhost:27017")
    db = client["spo"]
    collection = db["test"]
    for i in range(batch):
        part_x = test_x[batch_size * i: batch_size * (i + 1)]
        part_y = test_y[batch_size * i: batch_size * (i + 1)]

        record_list = []
        for j in range(batch_size):
            try:
                record_list.append({"x": part_x[j], "y": part_y[j]})
            except:
                break

        if record_list:
            collection.insert_many(record_list)
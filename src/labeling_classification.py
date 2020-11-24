from coco_nlp.embeddings import ALBertEmbedding, RoBertAEmbedding
from coco_nlp.tasks.labeling import BiLSTM_CRF_Model
from coco_nlp.generators import CorpusGenerator, MongoGenerator
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

# use default generator
def split_word_tag(wt_list):
    """
    Input: ["a O", "b O", "c B-SUBJ", "d I-SUBJ", "e O"]
    Output: [["a", "b", "c", "d", "e"], ["O"， “O”, "B-SUBJ", "I-SUBJ", "O"]]
    """
    wt_list = [wt.split(" ") for wt in wt_list]
    word_tuple, tag_tuple = list(zip(*wt_list))
    return [list(word_tuple), list(tag_tuple)]

def get_local_train_test_generator(train_bio_path, test_bio_path):
    with open(train_bio_path, "r", encoding="utf-8") as f:
        content = f.read()
        sentences = list(filter(lambda x: x, content.split("\n\n")))
        sentences = [split_word_tag(sen.split("\n")) for sen in sentences]
        train_x, train_y = list(zip(*sentences))

    with open(test_bio_path, "r", encoding="utf-8") as f:
        content = f.read()
        sentences = list(filter(lambda x: x, content.split("\n\n")))
        sentences = [split_word_tag(sen.split("\n")) for sen in sentences]
        test_x, test_y = list(zip(*sentences))

    train_gen = CorpusGenerator(train_x, train_y)
    test_gen = CorpusGenerator(test_x, test_y)
    return train_gen, test_gen

# use mongodb generator
def get_mongo_train_test_generator(mongo_url, db_name):
    train_gen = MongoGenerator(
        db_name=db_name,
        mongo_url=mongo_url,
        collection_name="train",
    )

    test_gen = MongoGenerator(
        db_name=db_name,
        mongo_url=mongo_url,
        collection_name="test",
    )

    return train_gen, test_gen


############################################## load model #############################################
num_threads = 2
tf.config.threading.set_inter_op_parallelism_threads(num_threads)
tf.config.threading.set_intra_op_parallelism_threads(num_threads)

model_path = "model/roberta_tiny"

albert_embedding = RoBertAEmbedding(model_folder=model_path)
bi_lstm_crf_model = BiLSTM_CRF_Model(
    embedding=albert_embedding,
    sequence_length=128,
)

train_gen, test_gen = get_mongo_train_test_generator(
    mongo_url="mongodb://localhost:27017",
    db_name="spo"
)

# train_gen, test_gen = get_local_train_test_generator("data/test_bio.txt", "data/test_bio.txt")

def save_checkpoint(checkout_path):
    return ModelCheckpoint(
        filepath=checkout_path,
        save_weights_only=True,
        verbose=1
    )

checkout_path = "model/spo/spo.ckpt"

# start training
bi_lstm_crf_model.fit_generator(
    train_sample_gen=train_gen,
    valid_sample_gen=test_gen,
    batch_size=64,
    epochs=10,
    callbacks=[
        save_checkpoint(checkout_path),
        EarlyStopping(monitor='loss', patience=2),
        TensorBoard(log_dir="log/spo")
    ]
)

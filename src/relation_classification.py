from coco_nlp.embeddings import ALBertEmbedding, RoBertAEmbedding
from coco_nlp.tasks.classification import BiLSTM_Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from coco_nlp.generators import MongoGenerator

train_gen = MongoGenerator(
    db_name="classification",
    mongo_url="mongodb://localhost:27017",
    collection_name="train",
)

test_gen = MongoGenerator(
    db_name="classification",
    mongo_url="mongodb://localhost:27017",
    collection_name="test",
)

model_path = "model/roberta_tiny"
albert_embedding = RoBertAEmbedding(model_folder=model_path)

bi_lstm_model = BiLSTM_Model(
    embedding=albert_embedding,
    sequence_length=128,
)

def save_checkpoint(checkout_path):
    return ModelCheckpoint(
        filepath=checkout_path,
        save_weights_only=True,
        verbose=1
    )

checkout_path = "model/classification/classification.ckpt"

bi_lstm_model.fit_generator(
    train_sample_gen=train_gen,
    valid_sample_gen=test_gen,
    batch_size=64,
    epochs=10,
    callbacks=[
        save_checkpoint(checkout_path),
        EarlyStopping(monitor='loss', patience=2),
        TensorBoard(log_dir="log/classification")
    ]
)

import tensorflow as tf
from tensorflow import keras
import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params,load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer

def create_model(max_seq_len, bert_ckpt_file,bert_config_file, classes):

    with tf.io.gfile.GFile(bert_config_file) as reader:
        bc = StockBertConfig.from_json_string(reader.read())
        bert_params = map_stock_config_to_params(bc)
        bert_params.adapter_size = None
        bert =  BertModelLayer.from_params(bert_params, name = "bert")

    input_ids = tf.keras.layers.Input(shape=(max_seq_len,), dtype='int32', name='input_')
    bert_output = bert(input_ids)

    print("BERT Shape {}".format(bert_output.shape))

    cls_out = tf.keras.layers.Lambda(lambda seq: seq[:,0,:])(bert_output)
    cls_out = tf.keras.layers.Dropout(0.5)(cls_out)
    logits = tf.keras.layers.Dense(units=768, activation='tanh')(cls_out)
    logits = tf.keras.layers.Dropout(0.5)(logits)
    logits = tf.keras.layers.Dense(units = len(classes), activation='softmax')(logits)

    model = tf.keras.Model(inputs = input_ids, outputs = logits)
    model.build(input_shape=(None, max_seq_len))
    load_stock_weights(bert, bert_ckpt_file)

    return model
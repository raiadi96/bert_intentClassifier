import os
import math
import datetime
from tqdm import tqdm

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params,load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import  rc

from utils import downloadBert, load_tokenizer
from data_loader import load_data
from DataModel import IntentDetectionData
from model import create_model

from sklearn.metrics import confusion_matrix, classification_report

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 8
RANDOM_SEED = 43

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

bert_url = "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip"

#Downloading BERT and Initializing the Tokenizer
bert_ckpt_dir, bert_ckpt_file, bert_config_file = downloadBert(bert_url)
tokenizer = load_tokenizer(os.path.join(bert_ckpt_dir,"vocab.txt"))

#load Dataset
train, test = load_data()

#Initialize the Train class
classes = train.intent.unique().tolist()
data =  IntentDetectionData(train, test, tokenizer, classes, max_seq_len = 128)

#initialise the model
model = create_model(data.max_seq_len, bert_ckpt_file,bert_config_file, classes)
print(model.summary())

model.compile(
  optimizer=keras.optimizers.Adam(1e-5),
  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
)

history = model.fit(
  x=data.train_x,
  y=data.train_y,
  validation_split=0.1,
  batch_size=16,
  shuffle=True,
  epochs=5
)
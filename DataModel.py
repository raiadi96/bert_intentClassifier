import bert
import numpy as np
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer
import tqdm

class IntentDetectionData:
    DATA_COLUMN = 'text'
    LABEL_COLUMN = 'intent'

    def __init__(self, train, test, tokenizer:FullTokenizer,classes, max_seq_len = 192):
        self.tokenizer = tokenizer
        self.max_seq_len = 0
        self.classes = classes
        ((self.train_x, self.train_y),(self.test_x, self.test_y)) = map(self.prepare_, [train, test])

        print("Max Seq Len {}".format(self.max_seq_len))
        self.max_seq_len = min(self.max_seq_len, max_seq_len)
        self.train_x, self.test_x = map(self._pad, [self.train_x, self.test_x])

    def prepare_(self, df):
        x,y = [], []
        for _, row in tqdm(df.iterrows()):
            text, label = row[IntentDetectionData.DATA_COLUMN], row[IntentDetectionData.LABEL_COLUMN]
            tokens = self.tokenizer.tokenize(text)
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            self.max_seq_len = max(self.max_seq_len, len(token_ids))
            x.append(token_ids)
            y.append(self.classes.index(label))
        return np.array(x), np.array(y)

    def _pad(self, ids):
        x = []
        for input_ids in ids:
            input_ids = input_ids[:min(len(input_ids), self.max_seq_len-2)]
            input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
            x.append(np.array(input_ids))
        return np.array(x)



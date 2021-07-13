import os
import wget
import zipfile
import tqdm
from bert.tokenization.bert_tokenization import FullTokenizer


BERT_DIR = "bert_files"
BERT_MODEL_NAME = "uncased_L-12_H-768_A-12"
def load_tokenizer(path):
    return FullTokenizer(vocab_file=path)

def downloadBert(path):
    if(os.path.isfile(BERT_MODEL_NAME + '.zip')):
        print("File already downloaded")
        pass
    else:
        print("#Download Started")
        files = wget.download(path)
        print("#Download Completed")
        with zipfile.ZipFile(files, 'r') as zip_ref:
            zip_ref.extractall()
    bert_ckpt_file = os.path.join(BERT_MODEL_NAME, "bert_model.ckpt")
    bert_config_file = os.path.join(BERT_MODEL_NAME, "bert_config.json")
    return BERT_MODEL_NAME, bert_ckpt_file, bert_config_file

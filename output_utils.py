import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import  rc
import datetime
import os
import pandas
import csv

OUTPUT_PATH = r"reports"

def set_defaults():
    sns.set(style='whitegrid', palette='muted', font_scale=1.2)

    HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

    sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

    rcParams['figure.figsize'] = 12, 8

def generate_confusion_matrix(df_cm):
    set_defaults()
    hmap = sns.heatmap(df_cm, annot =True, fmt = 'd')
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    file_name = "confusion_matrix-{}".format(datetime.datetime.utcnow().strftime("%m%d%Y%H%M%S"))
    plt.savefig(os.path.join(OUTPUT_PATH,file_name))
    print(os.path.join(OUTPUT_PATH, file_name))
    print("Confusion Matrix saved: {0}.Location: {1}".format(file_name, OUTPUT_PATH))


def generate_classification_report(classificaiton_report):
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    file_name = "classification_report-{}.csv".format(datetime.datetime.utcnow().strftime("%m%d%Y%H%M%S"))
    df = pandas.DataFrame(classificaiton_report).transpose()
    df.to_csv(os.path.join(OUTPUT_PATH,file_name))

def plot_train_history(history):
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    file_name = "loss_history-{}".format(datetime.datetime.utcnow().strftime("%m%d%Y%H%M%S"))
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'])
    plt.title('Loss over training epochs')
    plt.savefig(os.path.join(OUTPUT_PATH,file_name))

    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    file_name = "acc_history-{}".format(datetime.datetime.utcnow().strftime("%m%d%Y%H%M%S"))
    ax.plot(history.history['acc'])
    ax.plot(history.history['val_acc'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'])
    plt.title('Accuracy over training epochs')
    plt.savefig(os.path.join(OUTPUT_PATH,file_name))


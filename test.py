import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from output_utils import generate_confusion_matrix, generate_classification_report

def test_model(model, data, classes):
    y_pred = model.predict(data.test_x).argmax(axis=-1)
    cm = confusion_matrix(data.test_y, y_pred)
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)

    generate_confusion_matrix(df_cm)
    classificiation_report = classification_report(data.test_y, y_pred, target_names=classes, output_dict=True)
    generate_classification_report(classificiation_report)
    _, test_acc = model.evaluate(data.test_x, data.test_y)
    return test_acc



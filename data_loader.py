import pandas as pd
#importing data
def load_data():
    train = pd.read_csv("data/train.csv")
    valid = pd.read_csv("data/valid.csv")
    test = pd.read_csv("data/test.csv")
    train = train.append(valid).reset_index(drop=True)
    return train, test


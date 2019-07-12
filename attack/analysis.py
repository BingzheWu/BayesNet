import sklearn.metrics
import numpy as np
import pandas as pd
import sys


def log_map(x):
    import math
    return math.log(x)
def opsite(x):
    return 1-x

def get_average_thershold(critic_file):
    critics = pd.read_csv(critic_file)
    critics.columns = ['file_name', 'label', 'critic', 'membership']
    critics['critic'] = critics['critic'].map(opsite)
    train_sample = critics.loc[critics['membership']==1]
    test_sample = critics.loc[critics['membership']==0]
    print(train_sample['critic'].mean())
    print(test_sample['critic'].mean())
    gap = train_sample['critic'].mean()-test_sample['critic'].mean()
    print("gap%f"%gap)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(critics['membership'], critics['critic'], pos_label=1)
    ave = train_sample['critic'].mean()
    y_pred= [1 if x>ave else 0 for x in critics['critic']]
    f1_score = sklearn.metrics.f1_score(critics['membership'], y_pred, pos_label=1)
    average_precision_score = sklearn.metrics.average_precision_score(critics['membership'], critics['critic'], pos_label=1)
    auc = sklearn.metrics.auc(fpr, tpr)
    print("aucroc:%f"%auc)
    print("aucpr%f"%average_precision_score)
    print("f1@ave%f"%f1_score)

if __name__ == '__main__':
    critic_file = sys.argv[1]
    get_average_thershold(critic_file)

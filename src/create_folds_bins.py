import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
class ContinuousStratifiedKFold(StratifiedKFold):
    def split(selfself, x, y, groups=None):
        num_bins = int(np.floor(1 + np.log2(len(y))))
        bins = pd.cut(y, bins=num_bins, labels=False)
        return super().split(x, bins, groups)

if __name__ == "__main__":
    df = pd.read_csv("./data/train.csv")
    #df = df.dropna().reset_index(drop=True)
    df["kfold"] = -1

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    #kf = model_selection.KFold(n_splits=5)
    kf = ContinuousStratifiedKFold(n_splits=5) #shuffle=True, random_state=SEED

    for fold, (trn_, val_) in enumerate(kf.split(df, df.target.values)):
        print(len(trn_), len(val_))
        df.loc[val_, 'kfold'] = fold

    df.to_csv("./data/train_folds_bins.csv", index=False)
    


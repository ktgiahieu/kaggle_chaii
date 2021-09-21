import pandas as pd
from sklearn import model_selection


if __name__ == "__main__":
    df = pd.read_csv("./data/train_cleaned.csv")
    #df = df.dropna().reset_index(drop=True)
    df["kfold"] = -1

    df = df.sample(frac=1, random_state=50898).reset_index(drop=True)

    kf = model_selection.StratifiedKFold(n_splits=5)

    for fold, (trn_, val_) in enumerate(kf.split(X=df, y=df.language.values)):
        print(len(trn_), len(val_))
        df.loc[val_, 'kfold'] = fold

    df_external = pd.read_csv('/content/kaggle_chaii/data/external.csv')
    df_external["kfold"] = -1

    df = pd.concat([df, df_external])

    df.to_csv("./data/train_folds_external_cleaned.csv", index=False)
    

import pandas as pd
from sklearn import model_selection


if __name__ == "__main__":
    df = pd.read_csv("./data/train.csv")
    #df = df.dropna().reset_index(drop=True)
    df = df[df.language == 'hindi']
    df['len_answer'] = df.answer_text.apply(lambda x:len(x))
    df["kfold"] = -1

    df = df.sample(frac=1, random_state=50898).reset_index(drop=True)

    kf = model_selection.StratifiedKFold(n_splits=5)

    for fold, (trn_, val_) in enumerate(kf.split(X=df, y=df.len_answer.values)):
        print(len(trn_), len(val_))
        df.loc[val_, 'kfold'] = fold

    df.drop(columns=['len_answer'], inplace=True)
    df.to_csv("./data/train_folds_hindi.csv", index=False)
    

import pandas as pd
import json
from sklearn.model_selection import GroupShuffleSplit

def dataset_processing(json_file: json) ->  pd.core.frame.DataFrame:
    '''
    Preprocess dataset based on code vulnerability: each question has a corresponding safe and vulnerable code column, we split an entry into 2 entries
    Easier to then classify code as being vulnerable or not
    '''
    df = pd.read_json(json_file, lines=True)

    df_chosen = df[['question', 'chosen']].copy()
    df_chosen = df_chosen.rename(columns={'chosen': 'code'})
    df_chosen['isvuln'] = False

    df_rejected = df[['question', 'rejected', 'vulnerability']].copy()
    df_rejected = df_rejected.rename(columns={'rejected': 'code'})
    df_rejected['isvuln'] = True

    pdataset = pd.concat([df_chosen, df_rejected], ignore_index=True)

    # We want to have the vulnerable and correct code for the same question together for training the model
    # Hence we add a unique identifier to each entry with the same question
    pdataset['question_group_id'] = pd.factorize(pdataset['question'])[0]

    return pdataset


def shuffle_dataset(df: pd.core.frame.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Shuffle dataset so that correct code and vulnerable code for the same question remain in the same split (train/test)

    Args:
        -Unshuffled ataset
    Output:
        -Shuffle ready for model consumption
    '''

    groups = df["question_group_id"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    X = df.drop(columns=["is_vuln"])
    y = df["is_vuln"]

    for i, (train_index, test_index) in enumerate(splitter.split(X, y, groups)):
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}, group={groups[train_index]}")
        print(f"  Test:  index={test_index}, group={groups[test_index]}")

    train_df = df.iloc[train_index]
    test_df = df.iloc[test_index]
    
    return train_df, test_df
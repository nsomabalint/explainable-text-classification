import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from neattext.functions import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE


def load_sentiment_data():
    dataset_url = "https://drive.google.com/uc?export=download&id=1WTEViNH8i9a3ethzP5mN07evErguwgcq"
    dataset_df = pd.read_csv(dataset_url)
    return dataset_df[['tweet_id', 'airline_sentiment', 'airline_sentiment_confidence', 'airline', 'text']]



def split_stratified_into_train_val_test(df_input, stratify_colname='y',
                                         frac_train=0.6, frac_val=0.15, frac_test=0.25,
                                         random_state=None):

    # SOURCE: https://stackoverflow.com/a/60804119

    '''
    Splits a Pandas dataframe into three subsets (train, val, and test)
    following fractional ratios provided by the user, where each subset is
    stratified by the values in a specific column (that is, each subset has
    the same relative frequency of the values in the column). It performs this
    splitting by running train_test_split() twice.

    Parameters
    ----------
    df_input : Pandas dataframe
        Input dataframe to be split.
    stratify_colname : str
        The name of the column that will be used for stratification. Usually
        this column would be for the label.
    frac_train : float
    frac_val   : float
    frac_test  : float
        The ratios with which the dataframe will be split into train, val, and
        test data. The values should be expressed as float fractions and should
        sum to 1.0.
    random_state : int, None, or RandomStateInstance
        Value to be passed to train_test_split().

    Returns
    -------
    df_train, df_val, df_test :
        Dataframes containing the three splits.
    '''

    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' % \
                         (frac_train, frac_val, frac_test))

    if stratify_colname not in df_input.columns:
        raise ValueError('%s is not a column in the dataframe' % (stratify_colname))

    X = df_input # Contains all columns.
    y = df_input[[stratify_colname]] # Dataframe of just the column on which to stratify.

    # Split original dataframe into train and temp dataframes.
    df_train, df_temp, y_train, y_temp = train_test_split(X,
                                                          y,
                                                          stratify=y,
                                                          test_size=(1.0 - frac_train),
                                                          random_state=random_state)

    # Split the temp dataframe into val and test dataframes.
    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test, y_val, y_test = train_test_split(df_temp,
                                                      y_temp,
                                                      stratify=y_temp,
                                                      test_size=relative_frac_test,
                                                      random_state=random_state)

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

    return df_train, df_val, df_test


def make_classifiaction_data(dataset_df: pd.DataFrame):
    dataset_df = dataset_df.reset_index(drop=True)
    dataset_df['text'] = dataset_df['text'].apply(clean_text)
    
    vectorizer = TfidfVectorizer(stop_words='english', min_df=50, max_df=0.5)
    dataset_df['vectors'] = vectorizer.fit_transform(dataset_df['text'])[0]
    dataset_df['vectors'] = dataset_df['vectors'].apply(lambda v: v.toarray()[0])
    dataset_df['label_enc'] = dataset_df['airline_sentiment'].replace({"positive": 1, "neutral": 0, "negative":-1})
    
    train_df, dev_df, test_df = split_stratified_into_train_val_test(dataset_df, 'airline_sentiment')
    
    ds_memberships = []
    for ind in dataset_df.index:
        if ind in train_df.index:
            ds_memberships.append('train')
        elif ind in dev_df.index:
            ds_memberships.append('dev')
        else:
            ds_memberships.append('test')
    
    dataset_df['ds_split'] = ds_memberships
    
    train_x, train_y = train_df['vectors'].tolist(), train_df['label_enc'].tolist()
    dev_x, dev_y = dev_df['vectors'].tolist(), dev_df['label_enc'].tolist()
    test_x, test_y = test_df['vectors'].tolist(), test_df['label_enc'].tolist()
    
    res = {
        "train": (train_x, train_y),
        "dev": (dev_x, dev_y),
        "test": (test_x, test_y),
        "vectorizer": vectorizer,
        "df": dataset_df
    }
    
    return res


def get_model(classification_resources, resample=None):
    model = LogisticRegressionCV()
    
    train_x, train_y = classification_resources['train']

    if resample == 'SMOTE':
        sm = SMOTE(random_state=42)
        train_x, train_y = sm.fit_resample(train_x, train_y)
    elif resample == "DOWN_SAMPLE":
        df = classification_resources['df']
        df_train = df[df.ds_memberships == 'train']
        min_cls_membs = np.min([
            len(df_train[df_train.label_enc == -1]),
            len(df_train[df_train.label_enc == 0]),
            len(df_train[df_train.label_enc == 1])
            ])
        df_train_res = pd.concat([
                df_train[df_train.label_enc == -1].sample(min_cls_membs),
                df_train[df_train.label_enc == 0].sample(min_cls_membs),
                df_train[df_train.label_enc == 1].sample(min_cls_membs),
            ])
        train_x, train_y = df_train_res['vectors'].tolist(), df_train_res['label_enc'].tolist()
    
    model.fit(train_x, train_y)
    
    return model



def eval_model(model, classification_resources):
    test_x, test_y = classification_resources['test']
    
    preds = model.predict(test_x)
    print(classification_report(test_y, preds))


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, \
    f1_score
from sklearn.model_selection import train_test_split
import mlflow
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# Consts
CLASS_LABEL = 'MachineLearning'
train_df_path = 'data/train.csv'
test_df_path = 'data/test.csv'


def feature_engineering(raw_df):
    df = raw_df.copy()
    df['CreationDate'] = pd.to_datetime(df['CreationDate'])
    df['CreationDate_Epoch'] = df['CreationDate'].astype('int64') // 10 ** 9
    df = df.drop(columns=['Id', 'Tags'])
    df['Title_Len'] = df.Title.str.len()
    df['Body_Len'] = df.Body.str.len()
    # Drop the correlated features
    df = df.drop(columns=['FavoriteCount'])
    df['Text'] = df['Title'].fillna('') + ' ' + df['Body'].fillna('')
    return df


def fit_tfidf(train_df, test_df):
    tfidf = TfidfVectorizer(max_features=25000)
    tfidf.fit(train_df['Text'])
    train_tfidf = tfidf.transform(train_df['Text'])
    test_tfidf = tfidf.transform(test_df['Text'])
    return train_tfidf, test_tfidf, tfidf


def fit_model(train_X, train_y, test_X, test_y, random_state=42,register = True):

     with mlflow.start_run():
        
    
        
        tol = 0.001
        C = 1.2
        
        clf_tfidf = LogisticRegression( tol = tol, C = C, random_state=random_state)
        clf_tfidf.fit(train_X, train_y)

        train_metrics = eval_model(clf_tfidf, train_X, train_y)

        print("results on Train dataset")
        print(train_metrics)

        mlflow.log_metric("train roc_auc", train_metrics["roc_auc"])
        mlflow.log_metric("train average_precision", train_metrics["average_precision"])
        mlflow.log_metric("train accuracy", train_metrics["accuracy"])
        mlflow.log_metric("train precision", train_metrics["precision"])
        mlflow.log_metric("train recall", train_metrics['recall'])
        mlflow.log_metric("train f1", train_metrics["f1"])





        val_metrics = eval_model(clf_tfidf, test_X, test_y)


        mlflow.log_param("tol ", tol)
        mlflow.log_param("C ", C)

        mlflow.log_metric("val roc_auc", val_metrics["roc_auc"])
        mlflow.log_metric("val average_precision", val_metrics["average_precision"])
        mlflow.log_metric("val accuracy", val_metrics["accuracy"])
        mlflow.log_metric("val precision", val_metrics["precision"])
        mlflow.log_metric("val recall", val_metrics['recall'])
        mlflow.log_metric("val f1", val_metrics["f1"])

        print("results on Val dataset")
        print(val_metrics)

        remote_server_uri= "https://dagshub.com/mb16biswas/mlops101.mlflow"
        mlflow.set_tracking_uri(remote_server_uri)

        if(register):

            try:
                
                mlflow.sklearn.log_model(
                    clf_tfidf, "model", registered_model_name= "Logistic_Regression"
                )
            except Exception as e:

                print(e)
                mlflow.sklearn.log_model(clf_tfidf, "model")   
        
        else:

            mlflow.sklearn.log_model(clf_tfidf, "model")   

        return clf_tfidf


def eval_model(clf, X, y):
    y_proba = clf.predict_proba(X)[:, 1]
    y_pred = clf.predict(X)
    return {
        'roc_auc': roc_auc_score(y, y_proba),
        'average_precision': average_precision_score(y, y_proba),
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
    }


def split(random_state=42):
    print('Loading data...')
    df = pd.read_csv('data/CrossValidated-Questions.csv')
    df[CLASS_LABEL] = df['Tags'].str.contains('machine-learning').fillna(False)
    train_df, test_df = train_test_split(df, random_state=random_state, stratify=df[CLASS_LABEL])

    print('Saving split data...')
    train_df.to_csv(train_df_path)
    test_df.to_csv(test_df_path)


def train():

    print('Loading data...')
    train_df = pd.read_csv(train_df_path)
    test_df = pd.read_csv(test_df_path)

    print('Engineering features...')
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)

    print('Fitting TFIDF...')
    train_tfidf, test_tfidf, tfidf = fit_tfidf(train_df, test_df)

    print('Saving TFIDF object...')
    

    print('Training model...')
    train_y = train_df[CLASS_LABEL]
    model = fit_model(train_tfidf, train_y,test_tfidf, test_df[CLASS_LABEL])


if __name__  == "__main__" :
    
    train()
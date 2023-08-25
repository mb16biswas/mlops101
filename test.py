import datetime
from main import feature_engineering,  fit_tfidf
from sklearn.linear_model import LogisticRegression
import pandas as pd

train_df_path = 'data/train.csv'
def test():

    train_df = pd.read_csv(train_df_path).head(5)

    print(train_df)
    print(train_df.columns)
    print()


    train_df = feature_engineering(train_df)

    print(train_df)
    print(train_df.columns)
    print()

    train_tfidf, test_tfidf, tfidf = fit_tfidf(train_df, train_df)

    # print(train_tfidf)

    print(type(train_tfidf))



if __name__  == "__main__" :
    
    test()

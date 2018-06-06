# -*- coding: utf-8 -*-
from datetime import datetime, timedelta
from LightGBM import LightGBM
from JData import JData
from Feature import Feature
from datetime import date
from com_util import *
import metric_s
import numpy as np

def to_train(is_submit=False):
    # test code
    label_start_day = datetime(2017, 5, 1)
    label_end_day = datetime(2017, 5, 31)

    train_target_day = date(2017, 2, 1)
    valid_target_day = date(2017, 3, 1)
    test_target_day = date(2017, 4, 1)
    test_type='testscore'
    if is_submit:
        train_target_day = date(2017, 3, 1)
        valid_target_day = date(2017, 4, 1)
        test_target_day = date(2017, 5, 1)
        test_type='test'

    Jdata = JData('../input/', True)
    Jdata.loadData()

    #train data
    train_features = Feature(Jdata, label_start_day, label_end_day,train_target_day,type='train').generateFeature()

    #train data
    valid_features = Feature(Jdata, label_start_day, label_end_day,valid_target_day,type='valid').generateFeature()

    # test data
    test_features = Feature(Jdata, label_start_day, label_end_day,test_target_day,type=test_type).generateFeature()

    train = train_features.reset_index(drop=True)
    valid = valid_features.reset_index(drop=True)
    test = test_features.reset_index(drop=True)



    drop_features=["a_date","label_1","label_2"]

    #S1
    print(train.head())
    print(valid.head())
    print(test.head())
    train_x=train.drop(drop_features,axis=1).values
    valid_x=valid.drop(drop_features,axis=1).values
    drop_features.append('o_date_true')
    if is_submit == False:
        test_x = test.drop(drop_features, axis=1).values
    train_y=train["label_1"]
    valid_y=valid["label_1"]

    model = LightGBM(type='S1')
    model.fit(train_x,train_y,valid_x,valid_y)
    test["prob"] = model.predict(test_x)

    #S2
    train_y=train["label_2"]
    valid_y=valid["label_2"]

    model = LightGBM(type='S2')
    model.fit(train_x,train_y,valid_x,valid_y)
    test["pred_day_gap"] = model.predict(test_x)

    test["pred_date"]=list(map(lambda x,y:pred_date(x,y),test["a_date"],test["day_gap"]))
    test=Jdata.df_user_info[["user_id"]].merge(test,on="user_id",how="left").fillna(0)
    test=test.sort_values("prob",ascending=False).drop_duplicates("user_id",keep="first")

    if is_submit == False:
        # error_date

        test['error_gap']=(pd.to_datetime(test['pred_date'])-pd.to_datetime(test['o_date_true'])).dt.days
        s1= metric_s.s1(test)
        s2= metric_s.s2(test)
        s= metric_s.s(s1,s2)
        print(s)

    test=test[["user_id","prob","pred_date"]].copy()
    test[:50000][["user_id","pred_date"]].to_csv("sub.csv",index=None)

def main():
    to_train()

if __name__ == '__main__':
    main()
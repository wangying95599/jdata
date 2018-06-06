# -*- coding: utf-8 -*-
from BaseModel import BaseModel
import lightgbm as lgb

class LightGBM(BaseModel):
    def __init__(self, type='S1'):
        if type=='S1':
            self.params = {
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric': 'auc',
                'min_child_weight': 1.5,
                'num_leaves': 2 ** 5,
                'lambda_l2': 10,
                'subsample': 0.7,
                'colsample_bytree': 0.5,
                'colsample_bylevel': 0.5,
                'learning_rate': 0.1,
                'scale_pos_weight': 20,
                'seed': 2018,
                'nthread': 4,
                'silent': True,
            }
        else:
            self.params = {
                'boosting_type': 'gbdt',
                'objective': 'regression_l2',
                'metric': 'mse',
                'min_child_weight': 1.5,
                'num_leaves': 2 ** 5,
                'lambda_l2': 10,
                'subsample': 0.7,
                'colsample_bytree': 0.5,
                'colsample_bylevel': 0.5,
                'learning_rate': 0.1,
                'scale_pos_weight': 20,
                'seed': 2018,
                'nthread': 4,
                'silent': True,
            }

        self.num_round = 2000
        self.early_stopping_rounds = 100

    def process_data(self):
        pass

    def fit(self, train_x, train_y,valid_x,valid_y):
        train_matrix = lgb.Dataset(train_x, label=train_y)
        valid_matrix = lgb.Dataset(valid_x, label=valid_y)
        self.model = lgb.train(self.params, train_matrix, self.num_round, valid_sets=valid_matrix, early_stopping_rounds=self.early_stopping_rounds )

    def predict(self, test_x):
        return self.model.predict(test_x, num_iteration=self.model.best_iteration).reshape((test_x.shape[0], 1))


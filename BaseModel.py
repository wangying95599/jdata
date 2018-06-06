# -*- coding: utf-8 -*-


class BaseModel:
    def __init__(self, type):
        pass

    def process_data(self):
        pass

    def fit(self, train_x, train_y,valid_x,valid_y):
        pass

    def predict(self, test_x):
        pass

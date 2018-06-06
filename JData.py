# -*- coding: utf-8 -*-
import pandas as pd
import pandas as pd 
import numpy as np 
from datetime import datetime
import lightgbm as lgb
'''
数据的解析，不同业务，完全不一样，文件名，文件个数，文件内容都会不同，不必传入参数
文件路径可能传入不同，可以作为参数
'''
class JData(object):
	def __init__(self, path, load=False):
		self.path=path

		if load:
			self.loadData()

	def loadData(self):
		self.df_sku_info = pd.read_csv(self.path + "jdata_sku_basic_info.csv")
		self.df_user_info = pd.read_csv(self.path + "jdata_user_basic_info.csv")
		self.df_user_action = pd.read_csv(self.path + "jdata_user_action.csv")
		self.df_user_order = pd.read_csv(self.path + "jdata_user_order.csv")
		self.df_user_comment = pd.read_csv(self.path + "jdata_user_comment_score.csv")


		self.df_user_comment['c_datetime'] = pd.to_datetime(self.df_user_comment['comment_create_tm'])

		# sort by datetime
		self.df_user_action = self.df_user_action.sort_values(['user_id', 'a_date'])
		self.df_user_order = self.df_user_order.sort_values(['user_id', 'o_date'])
		self.df_user_comment = self.df_user_comment.sort_values(['user_id', 'c_datetime'])

		# change date2datetime
		self.df_user_action['a_date_pd'] = pd.to_datetime(self.df_user_action['a_date'])
		self.df_user_order['o_date_pd'] = pd.to_datetime(self.df_user_order['o_date'])
		self.df_user_comment['c_datetime'] = pd.to_datetime(self.df_user_comment['c_datetime'])
		# year month day
		self.df_user_order['o_date_year'] = self.df_user_order['o_date_pd'].dt.year
		self.df_user_order['o_date_month'] = self.df_user_order['o_date_pd'].dt.month
		self.df_user_order['o_date_day'] = self.df_user_order['o_date_pd'].dt.day

		self.df_user_action['a_date_year'] = self.df_user_action['a_date_pd'].dt.year
		self.df_user_action['a_date_month'] = self.df_user_action['a_date_pd'].dt.month
		self.df_user_action['a_date_day'] = self.df_user_action['a_date_pd'].dt.day

		self.df_user_comment['c_datetime_year'] = self.df_user_comment['c_datetime'].dt.year
		self.df_user_comment['c_datetime_month'] = self.df_user_comment['c_datetime'].dt.month
		self.df_user_comment['c_datetime_day'] = self.df_user_comment['c_datetime'].dt.day

		self.df_user_action = self.df_user_action.drop(['a_date_pd'],axis=1)
		self.df_user_order = self.df_user_order.drop(['o_date_pd'], axis=1)
		self.df_user_comment = self.df_user_comment.drop(['c_datetime'], axis=1)

if __name__ == '__main__':
    jdata = JData("../input/")
    jdata.loadData()
    print(jdata.df_user_comment.head(20))
    print(jdata.df_user_order.head(20))










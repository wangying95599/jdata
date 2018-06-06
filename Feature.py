# -*- coding: utf-8 -*-
import pandas as pd
import pandas as pd
import numpy as np
from datetime import datetime,date,timedelta
import JData
from com_util import *
import lightgbm as lgb
from plot_util import *
'''
特征
input：Jdata数据，日期范围，type=train/valid/test
'''
class Feature(object):
	def __init__(self, JData,
				label_start_day,
				label_end_day,
				target_day,
				type='train'):
		self.JData=JData
		self.label_start_day=label_start_day
		self.label_end_day=label_end_day
		self.type=type
		self.target_day=target_day

		# label columns
		self.LabelColumns = ['Label_30_101_BuyNum','Label_30_101_FirstTime']
		self.IDColumns = ['user_id']

	def generateFeature(self):
		jdata = self.JData

		sample_windows = 30  # 窗口大小
		label_windows = 30  # 窗口大小
		step = 31  # 滑窗步长
		slip_times = 0  # 滑窗次数


		slip_day = timedelta(days=step * slip_times)
		sample_start_day = self.target_day - timedelta(days=sample_windows) - slip_day
		sample_end_day = self.target_day - timedelta(days=1) - slip_day
		label_start_day = self.target_day - slip_day
		label_end_day = self.target_day + timedelta(days=label_windows) - slip_day

		sample_date = [str(d)[:10] for d in pd.date_range(sample_start_day, sample_end_day)]
		label_date = [str(d)[:10] for d in pd.date_range(label_start_day, label_end_day)]

		sample_date = pd.DataFrame({"a_date": sample_date})
		label_date = pd.DataFrame({"o_date": label_date})

		sample_start_day = self.target_day - timedelta(days=sample_windows) - slip_day
		sample_end_day = self.target_day - timedelta(days=1) - slip_day

		sample_date = [str(d)[:10] for d in pd.date_range(sample_start_day, sample_end_day)]
		train_date = sample_date.copy()
		label_date = [str(d)[:10] for d in pd.date_range(label_start_day, label_end_day)]

		sample_date = pd.DataFrame({"a_date": sample_date})
		label_date = pd.DataFrame({"o_date": label_date})
		# sql的内连接
		# print(jdata.df_user_action.head())
		sample = jdata.df_user_action.merge(sample_date, on="a_date", how="inner")
		# print(sample)
		sample = sample.merge(jdata.df_user_info, on="user_id", how="inner")
		sample = sample.merge(jdata.df_sku_info, on="sku_id", how="inner")

		# 关联order数据
		# o_date = pd.DataFrame({"o_date": train_date})
		# df_user_order = jdata.df_user_order.merge(o_date, on="o_date", how="inner")
		# sample = sample.merge(df_user_order, on=["sku_id","user_id"], how="inner")
		# sample["o_date_gap"] =  (pd.to_datetime(sample["o_date"]) - pd.to_datetime(sample["o_date"])).dt.days
		# sample = feat_min(sample, sample, ["user_id"], "o_date_gap")
		# sample = feat_mean(sample, sample, ["user_id"], "o_date_gap")
		# sample = sample.drop(['o_date'], axis=1)

		# 关联comment数据
		# sample = sample.merge(jdata.df_user_comment, on=["o_id","user_id"], how="left").fillna("-1")
		# sample=sample.drop(['comment_create_tm'],axis=1)
		# sample = feat_count(sample, sample, ["user_id"], "score_level")
		# plot_his(sample['age'])
		# plot_his(sample['sex'])

		self.df_Order_Comment_User_Sku = jdata.df_user_order. \
			merge(jdata.df_user_comment, on=['user_id', 'o_id'], how='left'). \
			merge(jdata.df_user_info, on='user_id', how='left'). \
			merge(jdata.df_sku_info, on='sku_id', how='left')
		# Action User Sku
		self.df_Action_User_Sku = jdata.df_user_action. \
			merge(jdata.df_user_info, on='user_id', how='left'). \
			merge(jdata.df_sku_info, on='sku_id', how='left')
		a_date = pd.DataFrame({"a_date": train_date})

		features_temp_Action_ = self.df_Action_User_Sku.merge(a_date, on="a_date", how="inner")
		# 用户浏览特征
		# sku_id cate 30 101 浏览数
		features_temp_ = features_temp_Action_[(features_temp_Action_['cate']==30) | (features_temp_Action_['cate']==101)].\
											groupby(['user_id'])['sku_id'].\
											nunique().\
											reset_index().\
											rename(columns={'user_id':'user_id','sku_id-1':'sku_id_cate_30_101_cnt'})
		#TODO?
		print(features_temp_.columns)
		sample = sample.merge(features_temp_,on=['user_id'],how='left')

		# a_date cate 30 101 天数
		features_temp_ = features_temp_Action_[(features_temp_Action_['cate']==30) | (features_temp_Action_['cate']==101)].\
											groupby(['user_id'])['a_date'].\
											nunique().\
											reset_index().\
											rename(columns={'user_id':'user_id','a_date-2':'a_date_cate_30_101_nuique'})
		sample = sample.merge(features_temp_,on=['user_id'],how='left')

		# 构造特征
		# sample = feat_mean(sample, sample, ["age"], "sex")
		sample['age_sex']=(sample['sex']+2)*(sample['age']+20)

		sample = feat_count(sample, sample, ["user_id"], "a_date")
		sample = feat_sum(sample, sample, ["user_id"], "a_num")
		sample = feat_max(sample, sample, ["user_id"], "a_num")
		sample = feat_nunique(sample, sample, ["user_id"], "sku_id")
		sample = feat_nunique(sample, sample, ["user_id"], "a_date")
		sample = feat_max(sample, sample, ["user_id"], "a_type")

		sample = feat_nunique(sample, sample, ["user_id"], "cate")
		sample = feat_nunique(sample, sample, ["user_id", "cate"], "sku_id")
		sample = feat_mean(sample, sample, ["user_id", "cate"], "price")

		sample["day_gap"] = (pd.to_datetime(self.target_day) - pd.to_datetime(sample["a_date"])).dt.days
		sample = feat_min(sample, sample, ["user_id"], "day_gap")
		sample = feat_mean(sample, sample, ["user_id"], "day_gap")

		# 去重，取最后一次浏览
		sample = sample.sort_values("a_date").drop_duplicates(["user_id"], keep="last")

		# 去重，取第一次购买
		label = jdata.df_user_order.merge(label_date, on="o_date", how="inner")[["user_id", "o_date"]].copy()
		label = label.sort_values("o_date").drop_duplicates(["user_id"], keep="first")

		if self.type == "train":
			sample = sample.merge(label, on=["user_id"], how="left").fillna("")
			sample["label_1"] = sample["o_date"].apply(lambda x: 1 if x else 0)
			sample["label_2"] = (pd.to_datetime(sample["o_date"]) - pd.to_datetime(sample["a_date"])).dt.days
			sample["label_2"] = sample["label_2"].fillna(100)
			del sample["o_date"]

		elif self.type == 'valid':
			sample = sample.merge(label, on=["user_id"], how="left").fillna("")
			sample["label_1"] = sample["o_date"].apply(lambda x: 1 if x else 0)
			sample["label_2"] = (pd.to_datetime(sample["o_date"]) - pd.to_datetime(sample["a_date"])).dt.days
			sample["label_2"] = sample["label_2"].fillna(100)
			del sample["o_date"]

		elif self.type == 'testscore':
			sample = sample.merge(label, on=["user_id"], how="left").fillna("")
			sample["o_date_true"] = sample["o_date"]
			sample["label_1"] = sample["o_date_true"].apply(lambda x: 1 if x else 0)
			sample["label_2"] = (pd.to_datetime(sample["o_date"]) - pd.to_datetime(sample["a_date"])).dt.days
			sample["label_2"] = sample["label_2"].fillna(100)
			del sample["o_date"]

		elif self.type == 'test':
			#5月份的数据，没有order
			sample["label_1"] = -1
			sample["label_2"] = -1

		print("            ",sample.columns)
		return sample


if __name__ == '__main__':
    label_start_day=datetime(2017,5,1)
    label_end_day=datetime(2017,5,31)

    Jdata = JData('../input/',True)
    feature= Feature(Jdata,label_start_day,label_end_day)



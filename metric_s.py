#encoding --utf-8--
import pandas as pd
import numpy as np

LABEL1='label_1'
LABEL2='error_gap'
def judge_w(df):
    if df[LABEL1]==1:
        return df.index
    else:
        return 0

def s1(df):
    print(df.head(100))
    print(df.info())

    df['w'] = 1 / (1 + np.log10(df.index + 1))

    s1_fm=df['w'].sum();
    print(s1_fm)

    df2 = df[df[LABEL1] == 1]  # 只留下对的。
    # print("df2   ",df2.head(100))
    s1_fz=df2['w'].sum();
    # print(s1_fz)

    s1_score=s1_fz/s1_fm
    print ("s1_score   ",s1_score)

    return s1_score

#前提是已经购买的。
def s2(df):
    df2 = df[df[LABEL1] == 1]


    df2['d2'] =  df2[LABEL2].apply(lambda x:10/(10+x*x))
    # print(df2.head())

    s2_fm=df2[LABEL1].count();
    # s2_fm=count
    # print(s2_fm)

    s2_fz=df2['d2'].sum()

    s2_score = s2_fz/s2_fm
    print("s2_score    ",s2_score)

    return s2_score


def s(s1,s2):
    a=0.4
    return a*s1+(1-a)*s2;




if __name__ == '__main__':
    # 验证数据有 是否购买的标签和间隔的标签。
    df = pd.read_csv('sub.csv')
    # df[LABEL1]=1
    # df[LABEL2]=5
    s1_score = s1(df)
    s2_score = s2(df)
    s_score = s(s1_score, s2_score)
    print(s_score)

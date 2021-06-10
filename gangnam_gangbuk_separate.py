import numpy as np
import pandas as pd

ori_data=pd.read_csv("airseoul_day.csv",index_col=[0],encoding='utf-8',engine='python')
#\print(ori_data.head())

nam=["강동구","송파구","강남구","서초구","관악구","동작구","영등포구","금천구","강서구","양천구","구로구"]
buk=["마포구","서대문구","은평구","강북구","성북구","종로구","중구","용산구","노원구","중랑구","동대문구","성동구","광진구"]
gu=["강동구","송파구","강남구","서초구","관악구","동작구","영등포구","금천구","강서구","양천구","구로구","마포구","서대문구","은평구","강북구","성북구","종로구","중구","용산구","노원구","중랑구","동대문구","성동구","광진구"]
daro=["강남대로","강변북로","공항대로","도산대로","동작대로","신촌로","영등포로","정릉로","종로","천호대로","청계천로","한강대로","홍릉로","화랑로"]

k,j=0,0
nam_day_sum_value=[]
buk_day_sum_value=[]
nam_mean=[]
buk_mean=[]
for i in range(len(ori_data)):
    if ori_data.iloc[i,1] in nam:
        value=list(ori_data.iloc[i,2:])
        k += 1

        if k==1:
            nam_day_sum_value=value
        else:
            nam_day_sum_value[0] += value[0]
            nam_day_sum_value[1] += value[1]
            nam_day_sum_value[2] += value[2]
            nam_day_sum_value[3] += value[3]
            nam_day_sum_value[4] += value[4]

        if k==len(nam):
            day_mean_value=[round(x/k,6) for x in nam_day_sum_value]
            day_mean_value.insert(0, ori_data.iloc[i, 0])
            day_mean_value.insert(1, "강남")
            nam_mean.append(day_mean_value)
            k=0

    if ori_data.iloc[i, 1] in buk:
        value = list(ori_data.iloc[i, 2:])
        j += 1

        if j == 1:
            buk_day_sum_value = value
        else:
            buk_day_sum_value[0] += value[0]
            buk_day_sum_value[1] += value[1]
            buk_day_sum_value[2] += value[2]
            buk_day_sum_value[3] += value[3]
            buk_day_sum_value[4] += value[4]

        if j == len(buk):
            day_mean_value = [round(x / j,6) for x in buk_day_sum_value]
            day_mean_value.insert(0,ori_data.iloc[i,0])
            day_mean_value.insert(1,"강북")
            buk_mean.append(day_mean_value)
            j = 0

print(len(nam_mean),len(buk_mean))
nam_df=pd.DataFrame(nam_mean,columns=ori_data.columns)
buk_df=pd.DataFrame(buk_mean,columns=ori_data.columns)
total_df=pd.concat([nam_df,buk_df])
total_df=total_df.reset_index(drop=True)
print(total_df.head())
total_df.to_csv("separate_SGG_NS.csv",encoding="utf-8-sig")
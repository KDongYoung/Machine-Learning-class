import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

ori_data=pd.read_csv("separate_SGG_NS.csv",index_col=[0],encoding='utf-8',engine='python')
print(ori_data.head())

# [강남 행수, 강북 행수]
ind=[0,0]
for i in range(len(ori_data)):
    if ori_data.iloc[i,1]=="강남":
        ind[0]+=1
    else:
        ind[1] += 1
print(ind)

# fig, ax=plt.subplots(2,5)
# ax[0][0].plot(ori_data.iloc[:3982,0],ori_data.iloc[:3982,2],label="NO2")
# ax[0][1].plot(ori_data.iloc[:3982,0],ori_data.iloc[:3982,3],label="O3")
# ax[0][2].plot(ori_data.iloc[:3982,0],ori_data.iloc[:3982,4],label="CO")
# ax[0][3].plot(ori_data.iloc[:3982,0],ori_data.iloc[:3982,5],label="SO2")
# ax[0][4].plot(ori_data.iloc[:3982,0],ori_data.iloc[:3982,6],label="PM10")
#
# ax[1][0].plot(ori_data.iloc[3982:,0],ori_data.iloc[3982:,2],label="NO2")
# ax[1][1].plot(ori_data.iloc[3982:,0],ori_data.iloc[3982:,3],label="O3")
# ax[1][2].plot(ori_data.iloc[3982:,0],ori_data.iloc[3982:,4],label="CO")
# ax[1][3].plot(ori_data.iloc[3982:,0],ori_data.iloc[3982:,5],label="SO2")
# ax[1][4].plot(ori_data.iloc[3982:,0],ori_data.iloc[3982:,6],label="PM10")
#
# ax[0][0].set_title("NO2")
# ax[0][1].set_title("O3")
# ax[0][2].set_title("CO")
# ax[0][3].set_title("SO2")
# ax[0][4].set_title("PM10")
# ax[1][0].set_title("NO2")
# ax[1][1].set_title("O3")
# ax[1][2].set_title("CO")
# ax[1][3].set_title("SO2")
# ax[1][4].set_title("PM10")
# plt.show()

### ACF, PACF 그림 그려서 lag 찾기
NO2=ori_data.iloc[:,2]
O3=ori_data.iloc[:,3]
CO=ori_data.iloc[:,4]
SO2=ori_data.iloc[:,5]
PM10=ori_data.iloc[:,6]

################################################################### N02
plot_acf(NO2)
plot_pacf(NO2)
plt.show()
# p=1, q=1

diff_1=NO2.diff(periods=1).iloc[1:]
diff_1.plot()
plot_acf(diff_1)
plot_pacf(diff_1)
plt.show()
# 차분 = 0

from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(NO2, order=(2,0,2))
model_fit = model.fit(trend='c',full_output=True, disp=1)
print(model_fit.summary())
model_fit.plot_predict()
plt.show()

fore = model_fit.forecast(steps=365)
print(fore)


################################################################### 03
plot_acf(O3)
plot_pacf(O3)
plt.show()
# p=1, q=1

diff_1=O3.diff(periods=1).iloc[1:]
diff_1.plot()
plot_acf(diff_1)
plot_pacf(diff_1)
plt.show()
# 차분 = 0

from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(O3, order=(2,0,2))
model_fit = model.fit(trend='c',full_output=True, disp=1)
print(model_fit.summary())
model_fit.plot_predict()
plt.show()

fore = model_fit.forecast(steps=365)
print(fore)


################################################################### CO
plot_acf(CO)
plot_pacf(CO)
plt.show()
# p=1, q=1

diff_1=CO.diff(periods=1).iloc[1:]
diff_1.plot()
plot_acf(diff_1)
plot_pacf(diff_1)
plt.show()
# 차분 = 0

from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(CO, order=(2,0,2))
model_fit = model.fit(trend='c',full_output=True, disp=1)
print(model_fit.summary())
model_fit.plot_predict()
plt.show()

fore = model_fit.forecast(steps=365)
print(fore)


################################################################### SO2
plot_acf(SO2)
plot_pacf(SO2)
plt.show()
# p=1, q=1

diff_1=CO.diff(periods=1).iloc[1:]
diff_1.plot()
plot_acf(diff_1)
plot_pacf(diff_1)
plt.show()
# 차분 = 0

from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(SO2, order=(2,0,2))
model_fit = model.fit(trend='c',full_output=True, disp=1)
print(model_fit.summary())
model_fit.plot_predict()
plt.show()
fore = model_fit.forecast(steps=365)
print(fore)

################################################################### PM10
plot_acf(PM10)
plot_pacf(PM10)
plt.show()
# p=1, q=1

diff_1=PM10.diff(periods=1).iloc[1:]
diff_1.plot()
plot_acf(diff_1)
plot_pacf(diff_1)
plt.show()
# 차분 = 0

from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(PM10, order=(2,0,2))
model_fit = model.fit(trend='c',full_output=True, disp=1)
print(model_fit.summary())
model_fit.plot_predict()
plt.show()

fore = model_fit.forecast(steps=365)
print(fore)

# 線性回歸--美國房價5000地區
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from sklearn.linear_model import LinearRegression as lr
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

file = "USA_Housing.csv"
data = pd.read_csv(file)
data2 = data.drop("Address",axis=1)
# 先瞭解資料標準差、中位數、平均數
print(data2.describe())
# print(data.info)
x = data2.drop("Price",axis = 1)
y = data["Price"]
# 完整x不需要有答案(實際價格)

lm = lr().fit(x,y)

print("尚未測試------")
coef = pd.DataFrame(lm.coef_,x.columns,columns=["迴歸係數"])
print(coef)
print()
print("訓練測試40%資料後結果------亂數種子設定為90")
xtrain,xtest,ytrain,ytest = tts(x,y,test_size=0.4,random_state=(100))

lt = lr().fit(xtrain,ytrain)

y_pred = lt.predict(xtest)

Mse_test = mse(ytest, y_pred)
Mae_test = mse(ytest,y_pred) ** 0.5

R_square = lt.score(xtest, ytest)
print("MSE: ",Mse_test) # 誤差值 近0越好
print("MAE: ",Mae_test)
print("R-squared: ",R_square) #R值0~1 近1越好
print("R值很接近1")
print("MSE離群值太大不看")

plt.figure(figsize=(10,6))
plt.scatter(ytest, y_pred)
plt.xlabel("實際價格")
plt.ylabel("預測價格")

plt.show()
#%%
import seaborn as sns
# 直方圖
sns.distplot(y)
# bar圖
sns.histplot(x="Price",data = data)
# corr()找出欄位間的關聯，會自動忽略空值 
# cmap ="BuPu"顏色展現方式
# annot數值是否顯示
sns.heatmap(data.corr() , annot = True , cmap = "BuPu" )

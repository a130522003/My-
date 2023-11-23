# 決策樹--辨認真鈔 False 0 True 1
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split as tts

file = "BanknoteAuth.csv"
data = pd.read_csv(file)

x = data.drop("Class",axis=1)

y = data["Class"]

xtrain,xtest,ytrain,ytest = tts(x,y,test_size=0.4,random_state=(90))

dtree = tree.DecisionTreeClassifier(max_depth=(4))
dtree.fit(xtrain, ytrain)

correct = dtree.score(xtest, ytest) * 100

print("在決策樹4層下測試40%資料結果")
print(f"準確率: {correct:.2f}%")
# print("預測值\n",dtree.predict(xtest))
# print("真實值\n",ytest.values)


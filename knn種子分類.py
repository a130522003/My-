# K-neighbors--種子分類
import  pandas as pd
from sklearn import neighbors as nb
from sklearn.model_selection import train_test_split as tts

file = "Seeds.csv"
data = pd.read_csv(file)

x = data.drop("Class",axis=1)
y = data["Class"]

xtrain,xtest,ytrain,ytest = tts(x,y,test_size=0.4,random_state=(90))
print("測試資料為40%，亂樹種子取90，找出準確率最高的K值")
lst = [i for i in range(1,10,2)]
accuracies = []
for k in lst:
    knn = nb.KNeighborsClassifier(n_neighbors=k)
    knn.fit(xtrain,ytrain)
    accuracies.append(knn.score(xtest, ytest))
    # accuracies.sort(reverse=True)
for k, accuracy in zip(lst, accuracies):
    print(f"K = {k}, 正確率 = {accuracy}")

print("k = 9時，準確率最高有94.05%")
    



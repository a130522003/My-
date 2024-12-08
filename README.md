機器學習專案彙總

本倉庫包含三個不同的機器學習專案，展示了多種監督學習技術的應用，涵蓋分類和回歸問題。這些專案展現了使用scikit-learn進行數據分析、模型訓練和評估的能力。
專案列表

1. 種子分類 (K-Nearest Neighbors)

技術: K近鄰算法 (KNN)
數據集: Seeds.csv
主要目標: 通過特徵進行種子分類

關鍵技術:

使用不同的K值進行交叉驗證
數據集切分：訓練集60%，測試集40%
最佳模型：K=9，準確率達94.05%

應用領域: 植物分類

2. 偽鈔檢測 (決策樹分類)

技術: 決策樹分類器
數據集: BanknoteAuth.csv
主要目標: 識別真鈔和偽鈔

關鍵技術:

決策樹深度限制為4層
資料集切分：訓練集60%，測試集40%
模型準確率：高達100%

應用領域: 金融安全、欺詐檢測

3. 美國房價預測 (線性回歸)

技術: 線性回歸
數據集: USA_Housing.csv
主要目標: 預測房屋價格

關鍵技術:

多變量線性回歸

模型評估指標：

MSE (均方誤差)
MAE (平均絕對誤差)
R-squared 值

數據可視化：實際vs預測價格散點圖
特徵相關性分析

應用領域: 房地產、金融分析

技術套件

Pandas
Scikit-learn
Matplotlib
Seaborn

運用技術

數據預處理
特徵選擇
模型訓練與評估
交叉驗證
數據可視化

總結
這些專案展示了機器學習在不同領域的應用，從農業分類到金融安全和房地產預測，體現了靈活運用各種機器學習算法解決實際問題的能力。
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib

# %%
# 讀取CSV檔案
data_train = pd.read_csv('train.csv')

# %%
# 從 JSON 檔案讀取 encoding_dict
with open('encoding_dict.json', 'r', encoding='utf-8') as f:
    encoding_dict = json.load(f)

# 遍歷所有 object 型別的欄位
for col in data_train.select_dtypes(include=['object']).columns:
    # 從 encoding_dict 中獲取對應欄位的編碼字典
    col_dict = encoding_dict[col]
    
    # 將欄位的內容轉換成對應的數值
    data_train[col] = data_train[col].map(col_dict)

# %%
# 去除ID欄位
data_train = data_train.drop(columns=['Id'])

# 目標變數是 'SalePrice'
X = data_train.drop(columns=['SalePrice'])
y = data_train['SalePrice']

# 從檔案讀取標準化器
scaler = joblib.load('scaler.pkl')

# 使用讀取的標準化器進行標準化
X_scaled = scaler.transform(X)

# 將資料拆分成訓練集和驗證集，比例為 80% 訓練集和 20% 驗證集，並將 numpy arrays 轉為 pandas DataFrames
X_train, X_val, y_train, y_val = train_test_split(pd.DataFrame(X_scaled), pd.DataFrame(y), test_size=0.2, random_state=42)

# %%
# 從 JSON 檔案讀取 best_params
with open('best_params.json', 'r', encoding='utf-8') as f:
    best_params = json.load(f)

# 顯示讀取後的 best_params
print(best_params)

# %%
# 創建隨機森林分類器
rf_model = RandomForestClassifier(**best_params)  # 使用 ** 解包參數字典

# 創建 StratifiedKFold 物件
kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=142)

# 初始化變數來儲存每次折疊的 MAE 和 RMSE
mae_scores = []
rmse_scores = []

# 進行 k 折交叉驗證
for train_index, val_index in kf.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    
    # 使用訓練集訓練模型
    rf_model.fit(X_train_fold, y_train_fold)
    
    # 使用驗證集進行預測
    y_pred_fold = rf_model.predict(X_val_fold)
    
    # 計算 MAE
    mae = mean_absolute_error(y_val_fold, y_pred_fold)
    mae_scores.append(mae)
    
    # 計算 RMSE
    rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred_fold))
    rmse_scores.append(rmse)

# 計算平均 MAE 和 RMSE
average_mae = sum(mae_scores) / len(mae_scores)
average_rmse = sum(rmse_scores) / len(rmse_scores)
print(f"Average Mean Absolute Error: {average_mae}")
print(f"Average Root Mean Squared Error: {average_rmse}")

# %%
# 將訓練好的模型儲存到檔案
joblib.dump(rf_model, 'random_forest_model_HousePrice.pkl')



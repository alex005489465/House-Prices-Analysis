import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# %%
# 讀取CSV檔案
data_train = pd.read_csv('train.csv')

# %%
# 手動轉換object到int64
# 創建一個字典來存儲每個欄位的內容與值
encoding_dict = {}

# 遍歷所有 object 型別的欄位
for col in data_train.select_dtypes(include=['object']).columns:
    # 獲取欄位的唯一值
    unique_values = data_train[col].unique()
    # 創建一個字典來存儲內容與值的對應關係
    col_dict = {value: idx for idx, value in enumerate(unique_values)}
    # 將這個字典添加到 encoding_dict 中
    encoding_dict[col] = col_dict
    # 將欄位的內容轉換成對應的數值
    data_train[col] = data_train[col].map(col_dict)

# 將 encoding_dict 存檔為 Json 檔案
with open('encoding_dict.json', 'w', encoding='utf-8') as f:
    json.dump(encoding_dict, f, ensure_ascii=False, indent=4)

# %%
# 去除ID欄位
data_train = data_train.drop(columns=['Id'])

# 目標變數是 'SalePrice'
X_train = data_train.drop(columns=['SalePrice'])
y_train = data_train['SalePrice']

# 對 X 進行標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 將標準化器儲存到檔案
joblib.dump(scaler, 'scaler.pkl')

# %%
# 定義參數網格，將 random_state 加入其中
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [10, 30, 50],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 4],
    'random_state': [131]
}

# 創建隨機森林分類器
rf_model = RandomForestClassifier()

# 創建 GridSearchCV 物件
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=4, scoring='neg_mean_absolute_error', n_jobs=-1)

# 進行網格搜索
grid_search.fit(X_train_scaled, y_train)

# 獲取最佳參數
best_params = grid_search.best_params_
print(f"Best parameters found: {best_params}")

# %%
# 將 best_params 存入 JSON 檔案
with open('best_params.json', 'w', encoding='utf-8') as f:
    json.dump(best_params, f, ensure_ascii=False, indent=4)




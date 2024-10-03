import pandas as pd
import json
import joblib


# %%
# 讀取CSV檔案
data_test = pd.read_csv('test.csv')

# %%
# 從 JSON 檔案讀取 encoding_dict
with open('encoding_dict.json', 'r', encoding='utf-8') as f:
    encoding_dict = json.load(f)

# 遍歷所有 object 型別的欄位
for col in data_test.select_dtypes(include=['object']).columns:
    # 如果欄位已經存在於編碼字典中，則使用現有的編碼字典
    if col in encoding_dict:
        col_dict = encoding_dict[col]
    else:
        col_dict = {}
        encoding_dict[col] = col_dict
    
    # 獲取欄位的唯一值
    unique_values = data_test[col].unique()
    
    # 檢查所有值是否都在字典中了，沒有在字典中的，加入字典中
    for value in unique_values:
        if value not in col_dict:
            col_dict[value] = len(col_dict)
    
    # 將欄位的內容轉換成對應的數值
    data_test[col] = data_test[col].map(col_dict)

# %%
# 將 Id 欄位取出並轉換成 numpy.ndarray 型別
id_array = data_test['Id'].values
# 從 data_test 中刪除 Id 欄位
data_test = data_test.drop(columns=['Id'])

# 從檔案讀取標準化器
scaler = joblib.load('scaler.pkl')

# 使用讀取的標準化器進行標準化
X_test_scaled = scaler.transform(data_test)

# %%
# 從檔案讀取模型
rf_model = joblib.load('random_forest_model_HousePrice.pkl')

# 使用測試資料進行預測
y_pred_test = rf_model.predict(X_test_scaled)
# 將 y_pred_test 轉換為 int64 型別
y_pred_test = y_pred_test.astype('int64')

# 將 id_array 和 y_pred_test 合併成 DataFrame
result = pd.DataFrame({'Id': id_array, 'SalePrice': y_pred_test})
# 將結果寫入 sample.csv 中
result.to_csv('test_predict.csv', index=False)



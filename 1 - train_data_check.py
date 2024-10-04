import pandas as pd

# %%
# 讀取CSV檔案
data_train = pd.read_csv('train.csv')

# 顯示前三行
print(data_train.head(3))

# %%
# 顯示DataFrame的形狀
print(data_train.shape, end='\n')

# 創建一個新的 DataFrame 來顯示欄位名稱和資料型別
columns_info = pd.DataFrame({
    'Column Name': data_train.columns,
    'Data Type': data_train.dtypes
})
print(columns_info)

# 統計每種資料型別的欄位數量
data_types_count = data_train.dtypes.value_counts()
print(data_types_count)

# 看資料的欄位資訊
print(data_train.info())

# 看數值欄位的統計值
print(data_train.describe())

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# %%
# 讀取CSV檔案
data_train_plot = pd.read_csv('train.csv')
data_test_predict_plot = pd.read_csv('test_predict.csv')

# 確保 SalePrice 欄位存在於資料中
if 'SalePrice' in data_train_plot.columns and 'SalePrice' in data_test_predict_plot.columns:
    plt.figure(figsize=(10, 6))
    
    # 繪製訓練集的 SalePrice 分布圖
    sns.histplot(data_train_plot['SalePrice'], bins=30, kde=True, color='blue', label='Training Data')
    
    # 繪製測試集的 SalePrice 分布圖
    sns.histplot(data_test_predict_plot['SalePrice'], bins=30, kde=True, color='red', label='Test Data')
          
    plt.title('SalePrice Distribution in Training and Test Data')
    plt.xlabel('SalePrice')
    plt.ylabel('Frequency')
    plt.legend()
    # 存成 PNG 檔案
    plt.savefig('SalePrice_Distribution.png')  
    plt.show()
    
else:
    print("SalePrice 欄位在資料中不存在")

# %%
# 讀取CSV檔案
data_train = pd.read_csv('train.csv')

# 刪除模型中沒有的列
data_train = data_train.drop(columns=['Id', 'SalePrice'])

# 從檔案讀取模型
rf_model = joblib.load('random_forest_model_HousePrice.pkl')

# 獲取特徵重要性
feature_importances = rf_model.feature_importances_

# 將特徵重要性轉換為 DataFrame
features = pd.DataFrame({
    'Feature': data_train.columns,
    'Importance': feature_importances
})

# 將Importance的數值限制在小數點以下第四位
features['Importance'] = features['Importance'].round(4)

# 按重要性排序
features = features.sort_values(by='Importance', ascending=False)

# 將特徵及其重要性存入 CSV 檔案
features.to_csv('feature_importances.csv', index=False)



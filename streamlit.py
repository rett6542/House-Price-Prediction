import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
plt.rcParams["font.family"] = ["Microsoft JHengHei"]
plt.rcParams["axes.unicode_minus"] = False #負數符號

st.title("預測房價的金額")
st.write("(採用隨機森林模型，準確率81%)")

#載入資料
house_price = pd.read_csv('housing.csv', encoding='big5')
house_price_old = house_price.copy()

#刪除空白值
house_price.dropna(inplace=True)

#將物件屬型資料轉為數值資料
label_encoder = preprocessing.LabelEncoder()
house_price["ocean_proximity_id"] = label_encoder.fit_transform(house_price["ocean_proximity"])

#設定數據集資料
X = house_price.drop(['ocean_proximity', 'median_house_value'], axis = 1)
y = house_price['median_house_value']

#切割數據
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)

#數據標準化(使用X_train的數據)
sc = preprocessing.StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#隨機森林(準確度82%)
#model_RF = RandomForestRegressor(criterion="absolute_error", n_estimators = 50)
# criterion : {"squared_error", "absolute_error", "poisson"
#model_RF.fit(X_train_std, y_train)

#讀入模型
with open ('model_RF_stored.pkl', 'rb') as model_file:
    model_RF = pickle.load(model_file)

pred_train = model_RF.predict(X_train_std)
pred_test = model_RF.predict(X_test_std)

MSE_train = np.mean((y_train-pred_train)**2)
MSE_test = np.mean((y_test-pred_test)**2)

MAE_train = np.mean(np.abs(y_train-pred_train))
MAE_test = np.mean(np.abs(y_test-pred_test))

R2_train = model_RF.score(X_train_std, y_train)
R2_test = model_RF.score(X_test_std, y_test)

evaluation = pd.DataFrame({"MSE":[MSE_train, MSE_test], "MAE":[MAE_train, MAE_test], "R-squared":[R2_train, R2_test]}, index=["train", "test"])
evaluation = evaluation.round(2)


# 預測房價金額
#建立一個輸入的表單
with st.form(key='my_form'):

    n_longitude = st.text_input("輸入經度")
    n_latitude = st.text_input("輸入緯度")
    n_housing_median_age = st.text_input("輸入房客年齡中位數")
    n_total_rooms = st.text_input("輸入總房間數")
    n_total_bedrooms = st.text_input("輸入總臥室數")
    n_population = st.text_input("輸入總人口數")
    n_households = st.text_input("輸入總家庭數")
    n_median_income = st.text_input("輸入收入中位數")
    n_ocean_proximity = st.selectbox("距離海洋位置",["靠近海灣", "距離海洋不到一小時", "內陸地區", "靠近海洋", "島嶼"])

    #建立一個按鍵做總匯入
    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    dict = {"靠近海灣":3, "距離海洋不到一小時":0, "內陸地區":1, "靠近海洋":4, "島嶼":2}
    n_ocean_proximity_id = dict.get(n_ocean_proximity)
    new_house_price = pd.DataFrame({
        "longitude": [n_longitude], "latitude": [n_latitude], "housing_median_age": [n_housing_median_age],
        "total_rooms": [n_total_rooms], "total_bedrooms": [n_total_bedrooms], "population": [n_population],
        "households ": [n_households], "median_income ": [n_median_income], "ocean_proximity_id": [n_ocean_proximity_id]})

    predicted_house_price = model_RF.predict(new_house_price)
    st.write("預測房價結果:")
    st.write(predicted_house_price)


if st.checkbox('顯示房價數據集資料'):
    house_price_old

if st.checkbox('顯示模型的評估數據'):
    evaluation

if st.checkbox('顯示模型的迴歸散佈圖'):

    fig1, ax1 = plt.subplots()
    fig1.set_size_inches(8, 8)
    ax1.scatter(y_test, pred_test)
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    st.pyplot(fig1)


if st.checkbox('顯示收入金額對人數直方圖'):

    fig2, ax2 = plt.subplots()
    fig2.set_size_inches(8, 8)
    sns.histplot(house_price['median_income'], bins=25, kde=True, color='skyblue', ax = ax2)
    ax2.set_title("收入金額對人數直方圖")
    ax2.set_xlabel("收入中位數")
    ax2.set_ylabel("人數")
    st.pyplot(fig2)

if st.checkbox('顯示近海位置對房價的盒鬚圖'):

    fig3, ax3 = plt.subplots()
    fig3.set_size_inches(8, 8)
    sns.boxplot(x='ocean_proximity', y='median_house_value', data=house_price, ax = ax3)
    ax3.set_title("近海位置對房價的盒鬚圖")
    ax3.set_xlabel("近海位置")
    ax3.set_ylabel("房價中位數")
    st.pyplot(fig3)

if st.checkbox('顯示各特徵之間的相關係數熱圖'):

    fig4, ax4 = plt.subplots()
    fig4.set_size_inches(8, 8)
    sns.heatmap(house_price.drop(columns=['ocean_proximity']).corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax = ax4)
    ax4.set_title('相關係數熱圖')
    st.pyplot(fig4)

#增加超連結
st.markdown("資料來源:[kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices)")



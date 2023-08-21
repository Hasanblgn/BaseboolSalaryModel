import pandas as pd
import numpy as np
from Modeling import helpers
import seaborn as sns
import matplotlib.pyplot as plt
from Modeling.BaseboolSalaryModel import helpers
import missingno as msno
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
pd.set_option("display.float_format", lambda x: "%.2f" % x)
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate

# AtBat 1986-1987 sezonunda bir beyzbol sopası ile topa yapılan vuruş sayısı
# Hits 1986-1987 sezonundaki isabet sayısı
# HmRun 1986-1987 sezonundaki en değerli vuruş sayısı
# Runs 1986-1987 sezonunda takımına kazandırdığı sayı
# RBI Bir vurucunun vuruş yaptıgında koşu yaptırdığı oyuncu sayısı
# Walks Karşı oyuncuya yaptırılan hata sayısı
# Years Oyuncunun major liginde oynama süresi (sene)
# CAtBat Oyuncunun kariyeri boyunca topa vurma sayısı
# CHits Oyuncunun kariyeri boyunca yaptığı isabetli vuruş sayısı
# CHmRun Oyucunun kariyeri boyunca yaptığı en değerli sayısı
# CRuns Oyuncunun kariyeri boyunca takımına kazandırdığı sayı
# CRBI Oyuncunun kariyeri boyunca koşu yaptırdırdığı oyuncu sayısı
# CWalks Oyuncun kariyeri boyunca karşı oyuncuya yaptırdığı hata sayısı
# League Oyuncunun sezon sonuna kadar oynadığı ligi gösteren A ve N seviyelerine sahip bir faktör
# Division 1986 sonunda oyuncunun oynadığı pozisyonu gösteren E ve W seviyelerine sahip bir faktör
# PutOuts Oyun icinde takım arkadaşınla yardımlaşma
# Assits 1986-1987 sezonunda oyuncunun yaptığı asist sayısı
# Errors 1986-1987 sezonundaki oyuncunun hata sayısı
# Salary Oyuncunun 1986-1987 sezonunda aldığı maaş(bin uzerinden)
# NewLeague 1987 sezonunun başında oyuncunun ligini gösteren A ve N seviyelerine sahip bir faktör

df = pd.read_csv("Modeling/BaseboolSalaryModel/hitters.csv")
df.head()


helpers.check_df(df)

# Öncelikle numerical değişkenlerimizi ve categorical değişkenlerimizi ayıralım

cat_cols, num_cols, cat_but_car = helpers.grab_col_names(df, cat_th=10, car_th=10)


for col in num_cols:
    helpers.target_count_with_num(df, target = "League", num_cols = col)


# Buradan inceledikten sonra outlier değerlerimize bakalım

for col in num_cols[:3]:
    plt.title(col)
    sns.boxplot(x=df[col])
    plt.show(block=True)


# Check Outliers
for col in num_cols:
    print(f"Outlier {col} :",helpers.check_outlier(df, col, q1=0.05, q3= 0.95))

# outlier görmek istersek.
for col in num_cols:
    print(f"{col} değişkenin low limit ve up limit değeri: ", helpers.outlier_thresholds(df, col, q1=0.05, q3 = 0.95))

# check edelim
for col in num_cols:
    print(f"{col} kolonunda outlier değeri var mı :", helpers.check_outlier(df, col_name=col, q1=0.05, q3=0.95))

# Threshold değerlerimiz ile outlier değerlerimizi baskılamış olduk.
for col in num_cols:
    if helpers.check_outlier(df, col, q1=0.05, q3=0.95):
        helpers.replace_with_thresholds(df, col, q1=0.05, q3=0.95)

# Kategorik Değişkenlerin Analizi

for col in cat_cols:
    helpers.cat_summary(df, col, plot=True)

# Sayısal Değişken Analizi

for col in num_cols:
    helpers.num_summary(df, col, False)

# Hedef değişken analizi
for col in cat_cols:
    helpers.target_summary_with_cat(df, "Salary", col)


# Çok maaş alanların değerleri incelenmesi
df.loc[(df["Salary"] > 2000 ), :].describe().T

# Korelasyon analizi;

drop_list_corr = helpers.high_correlated_cols(df, plot=True, corr_th=.90)

# Missing Values

df.isnull().values.any()

df.isnull().sum()

# Boş değer olan değerler.
df[df.isnull().any(axis=1)].head()

(df.isnull().sum() / df.shape[0]).sort_values(ascending=False)

# 59 tane boş değerimiz var bu değerler target kolonda olduğu için yanlılıgı önlemek açısından drop edilir.


print("Droptan önce : ############### : ",df.shape)
df.dropna(inplace=True)
print("Droptan sonra : ############### : ",df.shape)

nw_num_cols = [col for col in num_cols if col not in ["Salary", "Years"]]

# +1 eklememizin sebebi bir new Feature Extraction işlemi yaptığımız zaman bölüm kısmına sorun yaratmaması
df[nw_num_cols] = df[nw_num_cols] + 1


cat_cols, num_cols, cat_but_car = helpers.grab_col_names(df)




plt.rcParams["figure.figsize"] = [4,4]
plt.figure().set_figwidth(15)
msno.bar(df)
plt.show(block=True)

# Feature Extraction

df["Runs_and_Years"] = df["Runs"] * df["Years"]
df["Walks_and_Years"] = df["Walks"] * df["Years"]
df["HmRun_and_CATBat"] = df["HmRun"] / df["CAtBat"]
df["Years_and_CHmRun"] = df["CHmRun"] / df["Years"]
df["PutOuts_and_CHmRun"] = df["PutOuts"] + df["CHmRun"]
df["Errors_and_CHits"] = df["CHits"] * df["Errors"]
df["CRuns_and_PutOuts"] = df["CRuns"] * df["PutOuts"]
df["CRBI_and_Years"] = df["CRBI"] / df["Years"]
df["RBI_and_Hits"] = df["RBI"] * df["Hits"]
df["RBI_and_Assists"] = df["RBI"] + df["Assists"]
df["Errors_and_PutOuts"] = df["PutOuts"] / df["Errors"]
df["Run_and_CRun"] = df["Runs"] / df["CRuns"]


df.loc[((df["League"] == "A") & (df["Runs"] >= 60)), "League_Runs"] = "upperScorerA"
df.loc[((df["League"] == "A") & (df["Runs"] < 60)), "League_Runs"] = "lowerScorerA"
df.loc[((df["League"] == "N") & (df["Runs"] < 50)), "League_Runs"] = "lowerScorerN"
df.loc[((df["League"] == "N") & (df["Runs"] >= 50)), "League_Runs"] = "upperScorerN"
df.loc[((df["League"] == "A") & (df["Errors"] >= 8)), "League_Errors"] = "UnconsciousPlayerA"
df.loc[(df["League"] == "A") & ((df["Errors"] < 8) & (df["Errors"] >= 4)), "League_Errors"] = "normalPlayerA"
df.loc[(df["League"] == "A") & ((df["Errors"] < 4) & (df["Errors"] >= 0)), "League_Errors"] = "consciousPlayerA"
df.loc[((df["League"] == "N") & (df["Errors"] >= 9)), "League_Errors"] = "UnconsciousPlayerN"
df.loc[((df["League"] == "N") & ((df["Errors"] < 9) & (df["Errors"] >= 4))), "League_Errors"] = "normalPlayerN"
df.loc[((df["League"] == "N") & ((df["Errors"] < 4) & (df["Errors"] >= 0))), "League_Errors"] = "consciousPlayerN"


[df.groupby(col_cat).agg({col: ["mean", "median"]}) for col in num_cols for col_cat in cat_cols if col != "Salary"]

cat_cols, num_cols, cat_but_car = helpers.grab_col_names(df, cat_th=10, car_th=10)


# Label Encoding
binary_cols = [col for col in df.columns if df[col].dtypes not in ["float64", "int64"] and df[col].nunique() == 2]

for col in binary_cols:
    df = helpers.label_encoder(df, col)


# Rare Encoding

helpers.rare_analyser(df, "Salary", 0.1)

# Rare_Encoder işlemi olacaktı lakin rare değerlerimiz mevcut değil.

## ONE-HOT ENCODİNG

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df = helpers.one_hot_encoder(df, ohe_cols, drop_first=True)
cat_cols, num_cols, cat_but_car = helpers.grab_col_names(df)
num_cols.remove("Salary")

# Scaler
scaler = RobustScaler()
for col in num_cols:
    df[col] = scaler.fit_transform(df[[col]])


# modeling

X = df.drop("Salary", axis=1)
y = df[["Salary"]]

# Holdout yöntemini yapıyoruz.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=12, test_size=0.2)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# Sabit bias

reg_model.intercept_

# Ağırlık weight, coefficient

reg_model.coef_

# Lineer Regression y = b + wx
# iç çarpım yapıyoruz ve intercept_ i ekliyoruz.
np.inner(X_train.iloc[4, :].values, reg_model.coef_) + reg_model.intercept_
y_train.iloc[4]

#Tahmin Etme Prediction


# Model Performans Parametresi RMSE Train

y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
#242.700 Train RMSE

# Train RKARE model açıklanabilirlik değeri
reg_model.score(X_train, y_train)
#0.7301 Train Score

# Test RMSE

y_pred_test = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred_test))
#249.123 RMSE Test

# Test RKARE

reg_model.score(X_test, y_test)
# 0.5329 Test RKARE

# Cross Validation Process çapraz doğrulama

np.mean(np.sqrt(-cross_val_score(reg_model, X, y, cv=10, scoring = "neg_mean_squared_error")))
# 289.133 veri setimizde az deger oldugu icin butun degerleri kullandik cross validation isleminde

#smap = mean_absolute_error(y_train, y_pred) / y_train.mean()
mean_absolute_error(y_train, y_pred) / y_train.mean() # %32 sapma



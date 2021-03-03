#######################################
# LightGBM: Salary Predict
#######################################

# Veri Seti Hikayesi
"""
AtBat: 1986-1987 sezonunda bir beyzbol sopası ile topa yapılan vuruş sayısı
Hits: 1986-1987 sezonundaki isabet sayısı
HmRun: 1986-1987 sezonundaki en değerli vuruş sayısı
Runs: 1986-1987 sezonunda takımına kaç sayı kazandırdı
RBI: Bir vurucunun vuruş yaptıgında kaç tane oyuncuya koşu yaptırdığı.
Walks: Karşı oyuncuya kaç defa hata yaptırdığı
Years: Oyuncunun major liginde kaç sene oynadığı
CAtBat: Oyuncunun kariyeri boyunca kaç kez topa vurduğu
CHits: Oyuncunun kariyeri boyunca kaç kez isabetli vuruş yaptığı
CHmRun: Oyucunun kariyeri boyunca kaç kez en değerli vuruşu yaptığı
CRuns: Oyuncunun kariyeri boyunca takımına kaç tane sayı kazandırdığı
CRBI: Oyuncunun kariyeri boyunca kaç tane oyuncuya koşu yaptırdığı
CWalks: Oyuncun kariyeri boyunca karşı oyuncuya kaç kez hata yaptırdığı
League: Oyuncunun sezon sonuna kadar oynadığı ligi gösteren A ve N seviyelerine sahip bir faktör
Division: 1986 sonunda oyuncunun oynadığı pozisyonu gösteren E ve W seviyelerine sahip bir faktör
PutOuts: Oyun icinde takım arkadaşınla yardımlaşma
Assits: 1986-1987 sezonunda oyuncunun yaptığı asist sayısı
Errors: 1986-1987 sezonundaki oyuncunun hata sayısı
Salary: Oyuncunun 1986-1987 sezonunda aldığı maaş(bin uzerinden)
NewLeague: 1987 sezonunun başında oyuncunun ligini gösteren A ve N seviyelerine sahip bir faktör
"""

# Gerekli kütüphanelerin import edilmesi
import warnings
import pandas as pd
import missingno as msno
from lightgbm import LGBMRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import *
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from helpers.data_prep import *
from helpers.eda import *


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
pd.set_option('display.max_columns', None)

df = pd.read_csv("datasets/hitters.csv")
df.head()

# Aykırı değer varsa görebilmek için
msno.bar(df)
plt.show()

# veri seti gözlemler hakkında inceleme yapıldı

for i in ["object", "float", "integer", "bool"]:
    print(i.capitalize() + " Variables:", "\n", "# of Variables:",
          len(df.select_dtypes(i).columns), "\n",
          df.select_dtypes(i).columns.tolist(), "\n")


check_df(df)
missing_values_table(df)

# eksik gözlem içeren değişkenin görselleştirilmesi
sns.distplot(df.Salary)
plt.show()

# bağımsız değişken incelemesi
df["Salary"].describe()
df.dropna(inplace=True)
check_df(df["Salary"])

#######################################
# Feature Engineering
#######################################

# değişkenler arasındaki ilişkinin görselleştirilmesi
fig, ax = plt.subplots(figsize=(8,5))
sns.heatmap(df.corr(), annot=True, fmt=".2f", linewidths=0.5, ax=ax)
plt.show()

df["RateCRun"] = df["CHmRun"] / df["CRuns"]
df["RateRun"] = df["HmRun"] / df["Runs"]
# df["RateHit"] = df["Hits"] / df["AtBat"]
df["RateCHit"] = df["CHits"] / df["CAtBat"]

df['NEW_CWalks_CRuns'] = df['CWalks'] * df['CRuns']
df['NEW_CWalks_CAtBat'] = df['CWalks'] * df['CAtBat']
df['NEW_CWalks_CHits'] = df['CWalks'] * df['CHits']
df['NEW_CWalks_CRBI'] = df['CWalks'] * df['CRBI']


# df["Average_CAtBatYear"] = df["CAtBat"] / df["Years"]
df["Average_CHitsYear"] = df["CHits"] / df["Years"]
df["Average_CRunsYear"] = df["CRuns"] / df["Years"]
df["Average_CHmRunYear"] = df["CHmRun"] / df["Years"]
df["Average_CRBIYear"] = df["CRBI"] / df["Years"]
df["Average_CWalksYear"] = df["CWalks"] / df["Years"]

df.loc[(df["Years"] <= 3), "Experience"] = "Basic"
df.loc[(df["Years"] > 3) & (df["Years"] <= 8), "Experience"] = "Beginning"
df.loc[(df["Years"] > 8) & (df["Years"] <= 13), "Experience"] = "Normal"
df.loc[(df["Years"] > 13) & (df["Years"] <= 18), "Experience"] = "Experienced"
df.loc[(df["Years"] > 18), "Experience"] = "Vet"

df['Avg_CRBI_CAtBat'] = df['CRBI'] * df['CAtBat']
df['Avg_HmRun_RBI'] = df['HmRun'] * df['RBI']
df['Avg_Runs_AtBat'] = df['Runs'] / df['AtBat']
df['Avg_Runs_Hits'] = df['Runs'] / df['Hits']
df['Assists_Error'] = df['Assists'] / df['Errors']

df.columns = [col.upper() for col in df.columns]

df.drop(["YEARS", "HITS", "RUNS", "CHITS", "CRUNS", "ATBAT", "CATBAT", "CHMRUN", "HMRUN", "RBI", "CRBI"], inplace=True, axis=1)

df.shape
check_df(df)
missing_values_table(df)

# eksik gözlemleri median değeri ile doldurdum

nan_cols = ["ASSISTS_ERROR", "RATERUN"]

for col in nan_cols:
    df[col].fillna(df[col].median(), inplace=True)

df.isnull().sum()

# Ağaç yöntemlerinde aykırı değerin önemi kalmamaktadır.

#############################################
# ONE-HOT ENCODING
###########################################
ohe_cols = [col for col in df.columns if 10 >= len(df[col].unique()) > 2]
from helpers.data_prep import one_hot_encoder
df = one_hot_encoder(df, ohe_cols, drop_first=True)
df.columns = [col.upper() for col in df.columns]

#############################################
# LABEL ENCODING
###########################################
binary_cols = [col for col in df.columns if len(df[col].unique()) == 2 and df[col].dtypes == 'O']
from helpers.data_prep import label_encoder

for col in binary_cols:
    df = label_encoder(df, col)

df.columns = [col.upper() for col in df.columns]

df.shape
df.isnull().sum()
df.info()

#######################################
# LightGBM: Model & Tahmin
#######################################

y = df["SALARY"]
X = df.drop(["SALARY"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

lgb_model = LGBMRegressor().fit(X_train, y_train)
y_pred = lgb_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


#######################################
# Model Tuning
#######################################

lgb_model = LGBMRegressor()
lgb_model.get_params()


lgbm_params = {"learning_rate": [0.001, 0.01, 0.05, 0.1],
               "n_estimators": [500, 750, 1000, 2500],
               "max_depth": [-1, 2, 5, 8],
               "colsample_bytree": [1, 0.50, 0.75, 0.5],
               "num_leaves": [25, 31, 44]}


lgbm_cv_model = GridSearchCV(lgb_model,
                             lgbm_params,
                             cv=10,
                             n_jobs=-1,
                             verbose=2).fit(X_train, y_train)


lgbm_cv_model.best_params_

#######################################
# Final Model
#######################################

lgbm_tuned = LGBMRegressor(**lgbm_cv_model.best_params_).fit(X_train, y_train)
y_pred = lgbm_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

#######################################
# Feature Importance
#######################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(lgbm_tuned, X_train)


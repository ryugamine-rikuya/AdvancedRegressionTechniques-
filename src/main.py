import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score

import logging
from logging import getLogger, StreamHandler, Formatter, FileHandler


TRAINPATH = "../input/train.csv"
TESTPATH = "../input/test.csv"


#### logging setting ####
APPLICATIONNAME = "KaggleApplication"
logger = getLogger(APPLICATIONNAME)
logger.setLevel(logging.DEBUG)
handler_format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#### logging streamhandler ####
stream_handler = StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)

#### logging filehandler ####
file_handler = FileHandler('../log/'+APPLICATIONNAME+'.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(handler_format)
logger.addHandler(file_handler)



def readCSV(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        logger.error(e)
        return False

def main():

	#### import data ####
	logger.info("import data...")
	df_train = readCSV(TRAINPATH)
	df_test = readCSV(TESTPATH)
	df_train_index = df_train["Id"]
	df_test_index = df_test["Id"]
	Y = df_train["SalePrice"].values
	df_train.drop(["Id", "SalePrice"], axis=1, inplace=True)
	df_test.drop("Id", axis=1, inplace=True)

	df_all = pd.concat((df_train,df_test), sort=False).reset_index(drop=True)
	logger.info("train data : "+str(df_train.shape))
	logger.info("test data : "+str(df_test.shape))

	#### fill NaN ####
	df_all["PoolQC"].fillna('NA', inplace=True)
	df_all["MiscFeature"].fillna('None', inplace=True)
	df_all["Alley"].fillna('NA', inplace=True)
	df_all["Fence"].fillna('NA', inplace=True)
	df_all["FireplaceQu"].fillna('NA', inplace=True)
	df_all["GarageQual"].fillna('NA', inplace=True)
	df_all["GarageFinish"].fillna('NA', inplace=True)
	df_all["GarageCond"].fillna('NA', inplace=True)
	df_all["GarageType"].fillna('NA', inplace=True)
	df_all["BsmtCond"].fillna('NA', inplace=True)
	df_all["BsmtExposure"].fillna('NA', inplace=True)
	df_all["BsmtQual"].fillna('NA', inplace=True)
	df_all["BsmtFinType2"].fillna('NA', inplace=True)
	df_all["BsmtFinType1"].fillna('NA', inplace=True)
	df_all["MasVnrType"].fillna('None', inplace=True)
	df_all["GarageYrBlt"].fillna(0, inplace=True) # ガレージ築年数を0にするのも不思議な気はしますが、そもそもガレージがないので他に妥当な数字が思いつかず。
	df_all["MasVnrArea"].fillna(0, inplace=True)
	df_all["BsmtHalfBath"].fillna(0, inplace=True)
	df_all["BsmtFullBath"].fillna(0, inplace=True)
	df_all["TotalBsmtSF"].fillna(0, inplace=True)
	df_all["BsmtUnfSF"].fillna(0, inplace=True)
	df_all["BsmtFinSF2"].fillna(0, inplace=True)
	df_all["BsmtFinSF1"].fillna(0, inplace=True)
	df_all["GarageArea"].fillna(0, inplace=True)
	df_all["GarageCars"].fillna(0, inplace=True)
	df_all["MSZoning"].fillna('RL', inplace=True)
	df_all["Functional"].fillna('Typ', inplace=True)
	df_all["Utilities"].fillna("AllPub", inplace=True)
	df_all['SaleType'] = df_all['SaleType'].fillna(df_all['SaleType'].mode()[0])
	df_all['Exterior2nd'] = df_all['Exterior2nd'].fillna(df_all['Exterior2nd'].mode()[0])
	df_all['Exterior1st'] = df_all['Exterior1st'].fillna(df_all['Exterior1st'].mode()[0])
	df_all['KitchenQual'] = df_all['KitchenQual'].fillna(df_all['KitchenQual'].mode()[0])
	df_all['Electrical'] = df_all['Electrical'].fillna(df_all['Electrical'].mode()[0])
	f = lambda x: x.fillna(x.mean())
	df_all["LotFrontage"] = df_all.groupby("Neighborhood")["LotFrontage"].transform(f)

	#### change dummy ####
	df_all = pd.get_dummies(df_all)

	ntrain = df_train.shape[0]
	X = df_all[:ntrain]
	test = df_all[ntrain:]

	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1234)


	lasso = Lasso()
	rf = RandomForestRegressor()
	svr = svm.SVR()
	lasso_parameters = {'alpha':[0.1, 0.5, 1]}
	rf_parameters= {'n_estimators':[100, 500, 2000], 'max_depth':[3, 5, 10]}
	svr_parameters = {'C':[1e-1, 1e+1, 1e+3], 'epsilon':[0.05, 0.1, 0.3]}
	lasso_gs = GridSearchCV(lasso, lasso_parameters)
	lasso_gs.fit(X_train,y_train)
	rf_gs = GridSearchCV(rf, rf_parameters)
	rf_gs.fit(X_train,y_train)
	svr_gs = GridSearchCV(svr, svr_parameters)
	svr_gs.fit(X_train,y_train)

	GridSearchCV(cv=None, error_score='raise',
	estimator=svm.SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
	kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False),
	fit_params=None, iid=True, n_jobs=1,
	param_grid={'C': [0.1, 10.0, 1000.0], 'epsilon': [0.05, 0.1, 0.3]},
	pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
	scoring=None, verbose=0)

	#ラッソ回帰
	y_pred = lasso_gs.predict(X_test)
	print("ラッソ回帰でのRMSE:",np.sqrt(mean_squared_error(y_test, y_pred)))
	#ランダムフォレスト
	y_pred2 = rf_gs.predict(X_test)
	print("ランダムフォレストでのRMSE:",np.sqrt(mean_squared_error(y_test, y_pred2)))
	#SVR
	y_pred3 = svr_gs.predict(X_test)
	print("SVRでのRMSE:",np.sqrt(mean_squared_error(y_test, y_pred3)))

	lasso_pred = lasso_gs.predict(test)
	rf_pred = rf_gs.predict(test)
	svr_pred = svr_gs.predict(test)

	submission = pd.concat((df_test_index, pd.DataFrame(lasso_pred)), axis=1)
	submission.columns = ['Id', 'SalePrice']
	submission.to_csv("lasso_submission2.csv",sep=',',index=False)


	submission = pd.concat((df_test_index, pd.DataFrame(rf_pred)), axis=1)
	submission.columns = ['Id', 'SalePrice']
	submission.to_csv("rf_submission2.csv",sep=',',index=False)

	submission = pd.concat((df_test_index, pd.DataFrame(svr_pred)), axis=1)
	submission.columns = ['Id', 'SalePrice']
	submission.to_csv("svr_submission2.csv",sep=',',index=False)

	avg = (lasso_pred + rf_pred ) /2

	submission = pd.concat((df_test_index, pd.DataFrame(avg)), axis=1)
	submission.columns = ['Id', 'SalePrice']
	submission.to_csv("avg_submission2.csv",sep=',',index=False)





if __name__ == '__main__':
	main()

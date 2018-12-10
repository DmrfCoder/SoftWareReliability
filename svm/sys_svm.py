import numpy as np
import math
# from sklearn.linear_model.ftrl_classifier import FtrlClassifier
# from matplotlib import pylab as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.ensemble.forest import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time


def svr_main(X_train, Y_train, X_test, Y_test):
    clf = SVR(kernel='rbf', C=1e5, gamma=0.01)
    # clf.fit(X_train,Y_train)
    # y_pred = clf.predict(X_test)
    # plt.plot(X_test, y_pred, linestyle='-', color='red')

    # clf = GradientBoostingRegressor(n_estimators=100,max_depth=1)
    # clf = DecisionTreeRegressor(max_depth=25)
    # clf = ExtraTreesRegressor(n_estimators=2000,max_depth=14)
    # clf = xgb.XGBRegressor(n_estimators=2000,max_depth=25)
    # clf = RandomForestRegressor(n_estimators=1000,max_depth=26,n_jobs=7)
    predict_list = []
    TEST_SIZE=Y_test.shape[1]
    TRAIN_SIZE=Y_train.shape[1]
    Y_train=Y_train.reshape(-1,)
    Y_test=Y_test.reshape(-1,)
    X_train=X_train.T
    X_test=X_test.T
    clf.fit(X_train, Y_train)
    Y=np.concatenate((Y_train,Y_test))



    #predict_list=clf.predict(X_test)



    for i in range(TEST_SIZE):
        X = [[x] for x in range(0, TRAIN_SIZE + i)]
        clf.fit(X, Y[0:TRAIN_SIZE + i])

        y_pred = clf.predict([[TRAIN_SIZE + 1 + i]])
        predict_list.append(y_pred)

    print("mean_squared_error:%s" % mean_squared_error(Y_test, predict_list))
    print("sqrt of mean_squared_error:%s" % np.sqrt(mean_squared_error(Y_test, predict_list)))
    origin_data = Y_test
    print("origin data:%s" % origin_data)
    plt.plot([x for x in range(TRAIN_SIZE + 1, TRAIN_SIZE + TEST_SIZE + 1)], predict_list, linestyle='-', color='red',
             label='prediction model')
    plt.plot(X_test, Y_test, linestyle='-', color='blue', label='actual model')
    plt.legend(loc=1, prop={'size': 12})
    plt.show()




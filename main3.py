from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression


import bentoml

if __name__ == "__main__":

    X, y = make_regression(n_features=4, n_informative=2,random_state=0, shuffle=False)
                           
    regr = RandomForestRegressor(max_depth=2, random_state=0)
    regr.fit(X, y)


    lreg = linear_model.LinearRegression()
    lreg.fit(X, y)


    regr_model = bentoml.sklearn.save_model("random_reg", regr)
    print(f"Model saved: {regr_model}")

    lreg_model = bentoml.sklearn.save_model("linear_reg", lreg)
    print(f"Model saved: {lreg}")

    # Test running inference with BentoML runner

    reg_runner = bentoml.sklearn.get("random_reg:latest").to_runner()
    reg_runner.init_local()
    p = reg_runner.predict.run([[1, 1,1.2,3.7]])
    p1 = regr.predict([[1, 1,1.2,3.7]])
    print(p,p1)



    linear_runner = bentoml.sklearn.get("linear_reg:latest").to_runner()
    linear_runner.init_local()
    p = linear_runner.predict.run([[1, 1,1.2,3.7]])
    p1 = lreg.predict([[1, 1,1.2,3.7]])


    print(p,p1)





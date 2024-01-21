import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from ydata_profiling import ProfileReport
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from lazypredict.Supervised import LazyRegressor

data = pd.read_csv("StudentScore.xls")

# profile = ProfileReport(data, title="Student Report")
# profile.to_file("student_report.html")

target = "math score"
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

number_transformer = Pipeline( steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

education_values = ['some high school', 'high school', 'some college', "associate's degree", "bachelor's degree",
                    "master's degree"]
gender_values = ["male", "female"]
lunch_values = data["lunch"].unique()
test_values = data["test preparation course"].unique()
ordinal_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=[education_values, gender_values, lunch_values, test_values])),
])

# result = ordinal_transformer.fit_transform(x_train[["parental level of education"]])
#
# for i,j in zip(x_train[["parental level of education"]].values, result):
#     print("Before: {} - After: {}".format(i, j))

nominal_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(sparse=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num_features", number_transformer, ["reading score", "writing score"]),
    ("ord_features", ordinal_transformer, ["parental level of education", "gender", "lunch", "test preparation course"]),
    ("nom_features", nominal_transformer, ["race/ethnicity"]),
])

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    #("model", RandomForestRegressor()), #disable de chay lazy predict
])

x_train = model.fit_transform(x_train)
x_test = model.transform(x_test)

param_grid = {
    "model__n_estimators": [50, 100, 200],
    "model__criterion": ["squared_error", "absolute_error", "poisson"],
    "model__max_depth": [None, 2, 5],
    "preprocessor__num_features__imputer__strategy": ["median", "mean"]
}

# grid_search = GridSearchCV(estimator=model,
#                             param_grid=param_grid,
#                            scoring="neg_mean_squared_error", #scoring="r2",
#                            cv=5,
#                            n_jobs=-1,
#                            verbose=1)
# grid_search.fit(x_train, y_train)
#print("Best Estimator: ", grid_search.best_estimator_)
# print("Best Score: ", grid_search.best_score_)
# print("Best Params: ", grid_search.best_params_)

randomize_search = RandomizedSearchCV(estimator=model,
                            param_distributions=param_grid,
                           scoring="neg_mean_squared_error", #scoring="r2",
                           cv=5,
                           n_jobs=-1,
                           verbose=1,
                          n_iter=20)
# randomize_search.fit(x_train, y_train)
# y_predict = randomize_search.predict(x_test)
# print("Best Score: ", randomize_search.best_score_)
# print("Best Params: ", randomize_search.best_params_)

# model.fit(x_train, y_train)
# y_predict = model.predict(x_test)

# print("R2: {}".format(r2_score(y_test,y_predict)))
# print("MSE: {}".format(mean_squared_error(y_test, y_predict)))
# print("MAE: {}".format(mean_absolute_error(y_test, y_predict)))

# for i, j in zip(y_predict, y_test):
#     print("Predict: {} - Actual: {}".format(i, j))


clf = LazyRegressor(verbose=0,ignore_warnings=True)
models,predictions = clf.fit(x_train, x_test, y_train, y_test)



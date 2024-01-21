import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from ydata_profiling import ProfileReport

data = pd.read_csv("StudentScore.xls")
# corr = data.corr()
# profile = ProfileReport(data, title="My report")
# profile.to_file("student_report.html")
target = "math score"
x = data.drop(target, axis=1)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
education_values = ['some high school', 'high school', 'some college', "associate's degree", "bachelor's degree",
                    "master's degree"]
gender_values = ["male", "female"]
lunch_values = data["lunch"].unique()
test_values = data["test preparation course"].unique()
ord_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=[education_values, gender_values, lunch_values, test_values]))
])

nom_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(sparse=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num_features", num_transformer, ["reading score", "writing score"]),
    ("ord_features", ord_transformer, ["parental level of education", "gender", "lunch", "test preparation course"]),
    ("nom_features", nom_transformer, ["race/ethnicity"]),
])

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression()),
])

model.fit(x_train, y_train)
y_predict = model.predict(x_test)

for i, j in zip(y_predict, y_test):
    print("Prediction: {}. Actual value: {}".format(i, j))
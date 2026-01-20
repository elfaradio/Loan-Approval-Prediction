

import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,

)

import pickle

"""#Task-1"""

df = pd.read_csv('loan.csv')
df.shape

df.info()

df.head(5)

"""#Task-2

"""

df.isnull().sum()

dfc = df.copy()
dfc = dfc.drop('Loan_ID', axis=1)

# Encoding,Scaling,Missing values,outlier,feature engineering
numc = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']
for c in numc:
    dfc[c] = dfc[c].fillna(dfc[c].median())


catc = ['Gender', 'Married', 'Dependents', 'Self_Employed']
for c in catc:
    dfc[c] = dfc[c].fillna(dfc[c].mode()[0])


dfc['TotalIncome'] = dfc['ApplicantIncome'] + dfc['CoapplicantIncome']


X = dfc.drop('Loan_Status', axis=1)
y = dfc['Loan_Status'].map({'Y': 1, 'N': 0})


numf = X.select_dtypes(include=['int64', 'float64']).columns
catef = X.select_dtypes(include=['object']).columns


nump = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])


catp = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])


pre = ColumnTransformer(transformers=[
    ('num', nump, numf),
    ('cat', catp, catef)
])

"""#Task-3

"""

# PIpe line
rf_pipeline = Pipeline(steps=[
    ('preprocessor', pre),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    ))
])

"""#Task-4

We Selected RandomForestClassifier because it handles non-linear relation,also it need minimal prepropcessing,perform well on mediam dataseet,best for binary classification yes or no

#Task-5
"""

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf_pipeline.fit(X_train, y_train)
y_pred = rf_pipeline.predict(X_test)

"""#Task-6"""

# from google ,i searched google for scoring param in binary classification
score = cross_val_score(rf_pipeline, X_train, y_train,
                        cv=5, scoring='accuracy')
score

"""#Task-7"""

grid_param = {  # iused google for best param
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2],
    'classifier__bootstrap': [True, False]
}


gs = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=grid_param,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

gs.fit(X_train, y_train)

print("best param: ", gs.best_params_)
print("best   score: ", gs.best_score_)

rs = RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=grid_param,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)
rs.fit(X_train, y_train)


print("best param:", rs.best_params_)
print("best   score:", rs.best_score_)

"""#Task-8

"""

if gs.best_score_ >= rs.best_score_:
    # print("yes")
    model = gs.best_estimator_
else:
    model = rs.best_estimator

"""#Task-9"""

pred = model.predict(X_test)

print("Accuracy ", accuracy_score(y_test, pred))
print("\n")
print("Precision: ", precision_score(y_test, pred))
print("\n")
print("Recall :", recall_score(y_test, pred))
print("\n")
print("F1: ", f1_score(y_test, pred))
print("\n")
print("Reportv ", classification_report(y_test, pred))
print("\n")
print("CM ", confusion_matrix(y_test, pred))

"""#Save-Model"""

with open("loan_model.pkl", "wb") as f:
    pickle.dump(model, f)

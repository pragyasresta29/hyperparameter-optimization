from sklearn import tree, metrics, model_selection
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb


#DecisionTree Classifier
def dt_classifier(data, params):
    dtc = tree.DecisionTreeClassifier(**params)
    dtc.fit(data.data, data.target)
    cv = model_selection.cross_validate(dtc, data.data, data.target, scoring="accuracy", cv=10)
    print("---- Decision Tree Classifier----")
    print("Test AUC Roc Score: ", cv["test_score"].mean())
    print("--------------------------------")
    return cv["test_score"].mean()


#KNN classifier
def ada_classifier(data, params):
    ada = AdaBoostClassifier(**params)
    ada.fit(data.data, data.target)
    cv = model_selection.cross_validate(ada, data.data, data.target, scoring="accuracy", cv=10)
    print("---- Adaboost Classifier----")
    print("Test AUC Roc Score: ", cv["test_score"].mean())
    print("--------------------------------")
    return cv["test_score"].mean()


#XGBOOST
def xgboost_classifier(data, params):
    xg = xgb.XGBClassifier(**params, verbosity=0, silent=True)
    xg.fit(data.data, data.target)
    cv = model_selection.cross_validate(xg, data.data, data.target, scoring="accuracy", cv=10)
    print("---- Adaboost Classifier----")
    print("Test AUC Roc Score: ", cv["test_score"].mean())
    print("--------------------------------")
    return cv["test_score"].mean()


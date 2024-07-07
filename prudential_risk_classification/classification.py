import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ValidationCurveDisplay, LearningCurveDisplay
from sklearn.preprocessing import StandardScaler

train = pd.read_csv("./prudential-life-insurance-assessment/train.csv")
train_df = train.sample(n=10000)

categorical_variables = ["Product_Info_1","Product_Info_2","Product_Info_3","Product_Info_5","Product_Info_6","Product_Info_7","Employment_Info_2","Employment_Info_3","Employment_Info_5","InsuredInfo_1","InsuredInfo_2","InsuredInfo_3","InsuredInfo_4","InsuredInfo_5","InsuredInfo_6","InsuredInfo_7","Insurance_History_1","Insurance_History_2","Insurance_History_3","Insurance_History_4","Insurance_History_7","Insurance_History_8","Insurance_History_9","Family_Hist_1","Medical_History_2","Medical_History_3","Medical_History_4","Medical_History_5","Medical_History_6","Medical_History_7","Medical_History_8","Medical_History_9","Medical_History_11","Medical_History_12","Medical_History_13","Medical_History_14","Medical_History_16","Medical_History_17","Medical_History_18","Medical_History_19","Medical_History_20","Medical_History_21","Medical_History_22","Medical_History_23","Medical_History_25","Medical_History_26","Medical_History_27","Medical_History_28","Medical_History_29","Medical_History_30","Medical_History_31","Medical_History_33","Medical_History_34","Medical_History_35","Medical_History_36","Medical_History_37","Medical_History_38","Medical_History_39","Medical_History_40","Medical_History_41"]
continuous_variables = ["Product_Info_4","Ins_Age","Ht","Wt","BMI","Employment_Info_1","Employment_Info_4","Employment_Info_6","Insurance_History_5","Family_Hist_2","Family_Hist_3","Family_Hist_4","Family_Hist_5"]
discrete_variables = ["Medical_History_1","Medical_History_10","Medical_History_15","Medical_History_24","Medical_History_32"]

imp_categorical = SimpleImputer(strategy="most_frequent")
imp_continuous = SimpleImputer(strategy="median")
imp_discrete = SimpleImputer(strategy="median")

train_df[discrete_variables] = imp_discrete.fit_transform(train_df[discrete_variables])
train_df[categorical_variables] = imp_categorical.fit_transform(train_df[categorical_variables])
train_df[continuous_variables] = imp_continuous.fit_transform(train_df[continuous_variables])

# As the categorical data does not have meaningful numerical implications, the most frequent strategy makes the most sense.
# For continuous data, the median strategy is chosen as it ensures the filled values are not affected by outliers or skewed distributions.
# For discrete data, we use the median, as the data represents countable values and the median ensures that the filled values represent an actual data from the dataset and is not affected by outliers.

# Adding an additional boolean feature that indicates rows with missing values could be useful in some cases. 
# If the missing data provids an insight into the result, for example missing data could represent a correlation with other variables, this can now be captured by the new added variable, and used to train the model effectively. 
# However, we must also take into consideration that doing so increases the complexity of the model, or that adding an extra variable may not represent any insights and may not be optimal to be used in training.

train_df = pd.get_dummies(train_df, columns=categorical_variables)
scaled_x = StandardScaler().fit_transform(train_df.drop("Response", axis=1))

qwk_scorer = make_scorer(cohen_kappa_score, weights='quadratic')
qwk_scorer.__name__ = 'quadratic_weighted_kappa'

ValidationCurveDisplay.from_estimator(
    DecisionTreeClassifier(), scaled_x, train_df["Response"], param_name="max_depth", param_range=[d for d in range(2, 21)], scoring=qwk_scorer, cv=5, n_jobs=-1
)
plt.show()
# The best choice of max_depth from this plot is depth = 12
LearningCurveDisplay.from_estimator(
    DecisionTreeClassifier(max_depth=12), scaled_x, train_df["Response"], scoring=qwk_scorer, cv=5, n_jobs=-1
)
plt.show()

ValidationCurveDisplay.from_estimator(
    LogisticRegression(), scaled_x, train_df["Response"], param_name="C", param_range=np.logspace(0, 3, 50), scoring=qwk_scorer, cv=5, n_jobs=-1
)
plt.show()
# The best value of C from this plot is C = 1
LearningCurveDisplay.from_estimator(
    LogisticRegression(C=1), scaled_x, train_df["Response"], scoring=qwk_scorer, cv=5, n_jobs=-1
)
plt.show()
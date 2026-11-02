import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2

## NOTE: The data used in this demo comes from the UCI machine learning
## repository.
## http://archive.ics.uci.edu/ml/index.php
## Specifically, this is the heart disease data set.
## http://archive.ics.uci.edu/ml/datasets/Heart+Disease

df = pd.read_csv("processed_cleveland_data.csv", header=None)
# df = pd.read_csv("processed_cleveland_data.csv", header=None, na_values="?")

#####################################
##
## Reformat the df so that it is
## 1) Easy to use (add nice column names)
## 2) Interpreted correctly by the logistic regression..
##
#####################################
print(df.head())  # you see data, but no column names

df.columns = [
    "age",
    "sex",  # 0 = female, 1 = male
    "cp",  # chest pain
    # 1 = typical angina,
    # 2 = atypical angina,
    # 3 = non-anginal pain,
    # 4 = asymptomatic
    "trestbps",  # resting blood pressure (in mm Hg)
    "chol",  # serum cholestoral in mg/dl
    "fbs",  # fasting blood sugar if less than 120 mg/dl, 1 = TRUE, 0 = FALSE
    "restecg",  # resting electrocardiographic results
    # 1 = normal
    # 2 = having ST-T wave abnormality
    # 3 = showing probable or definite left ventricular hypertrophy
    "thalach",  # maximum heart rate achieved
    "exang",  # exercise induced angina, 1 = yes, 0 = no
    "oldpeak",  # ST depression induced by exercise relative to rest
    "slope",  # the slope of the peak exercise ST segment
    # 1 = upsloping
    # 2 = flat
    # 3 = downsloping
    "ca",  # number of major vessels (0-3) colored by fluoroscopy
    "thal",  # this is short of thalium heart scan
    # 3 = normal (no cold spots)
    # 6 = fixed defect (cold spots during rest and exercise)
    # 7 = reversible defect (when cold spots only appear during exercise)
    "hd",  # (the predicted attribute) - diagnosis of heart disease
    # 0 if less than or equal to 50% diameter narrowing
    # 1 if greater than 50% diameter narrowing
]

print(df.head(3))  # now we have data and column names

print(df.dtypes)  # "ca" and "thal" are read as strings because of the "?"s
# in the dataset
print(df["ca"].unique())
print(df["thal"].unique())

## First, convert "?"s to NAs...
df = df.replace("?", np.nan)

## "ca" and "thal" had "?"s in them, so they were read as strings. Convert them
## to numbers (NaN stays NaN)
df["ca"] = pd.to_numeric(df["ca"])
df["thal"] = pd.to_numeric(df["thal"])

## Now add factors (categoricals) for variables that are factors
df["sex"] = df["sex"].map({0: "F", 1: "M"})

## This replaces 0 and 1 with "Healthy" and "Unhealthy"
# print(df["hd"].unique())
# df["hd"] = np.where(df["hd"] == 0, "Healthy", "Unhealthy")
df["hd"] = np.where(df["hd"] == 0, 0, 1).astype(int)
# df.head()

## Now determine how many rows have "NA" (aka "Missing data"). If it's just
## a few, we can remove them from the dataset, otherwise we should consider
## imputing the values with a Random Forest or some other imputation method.
missing = df["ca"].isna() | df["thal"].isna()
print(missing.sum())
print(len(df))
print(missing.sum() / len(df))
## so 6 of the 303 rows of data have missing values. This isn't a large
## percentage (2%), so we can just remove them from the dataset
## NOTE: This is different from when we did machine learning with
## Random Forests. When we did that, we imputed values.
print(len(df))
df = df[~missing].copy()
print(len(df))

## Convert the factor columns to categoricals. Categories are sorted, so the
## first level is the reference level (just like the default in R).
## The numeric ones are converted from floats to integers first so the levels
## are "0", "1", ... and not "0.0", "1.0", ...
# for col in ["cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]:
#     df[col] = df[col].astype(int)
for col in ["cp", 
            "fbs", 
            "restecg", 
            "exang", 
            "slope", 
            "ca", 
            "thal"]:
    df[col] = df[col].astype(int)
# for col in ["sex", 
#             "cp", 
#             "fbs", 
#             "restecg", 
#             "exang", 
#             "slope", 
#             "ca", 
#             "thal", 
#             "hd"]:
#     df[col] = df[col].astype(str).astype("category")
for col in ["sex", 
            "cp", 
            "fbs", 
            "restecg", 
            "exang", 
            "slope", 
            "ca", 
            "thal"]:
    df[col] = df[col].astype(str).astype("category")

print(df.info())  ## this shows that the correct columns are categoricals

#####################################
##
## Now we can do some quality control by making sure all of the factor
## levels are represented by people with and without heart disease (hd)
##
## NOTE: We also want to exclude variables that only have 1 or 2 samples in
## a category since +/- one or two samples can have a large effect on the
## odds/log(odds)
##
##
#####################################
print(pd.crosstab(df["hd"], df["sex"]))
print(pd.crosstab(df["hd"], df["cp"]))
print(pd.crosstab(df["hd"], df["fbs"]))
print(pd.crosstab(df["hd"], df["restecg"]))
df = df.drop(columns="restecg")
print(df.head())
print(pd.crosstab(df["hd"], df["exang"]))
print(pd.crosstab(df["hd"], df["slope"]))
print(pd.crosstab(df["hd"], df["ca"]))
print(pd.crosstab(df["hd"], df["thal"]))

#####################################
##
## Now we are ready for some logistic regression. First we'll create a very
## simple model that uses sex to predict heart disease
##
#####################################

## let's start super simple and see if sex (female/male) is a good
## predictor...
## First, let's just look at the raw data...
print(pd.crosstab(df["hd"], df["sex"]))
#           sex
# hd         F   M
# Healthy    71  89
# Unhealthy  25 112
## Most of the females are healthy and most of the males are unhealthy.
## Being female is likely to decrease the odds in being unhealthy.
##    In other words, if a sample is female, the odds are against it that it
##    will be unhealthy
## Being male is likely to increase the odds in being unhealthy...
##    In other words, if a sample is male, the odds are for it being unhealthy

###########
##
## Now do the actual logistic regression
##
###########

## statsmodels needs a 0/1 outcome
logistic = smf.glm("hd ~ sex", data=df, family=sm.families.Binomial()).fit()
print(logistic.summary())
## Intercept   -1.0438     0.2326  -4.488 0.000
## sex[T.M]     1.2737     0.2725   4.674 0.000

## Let's start by going through the first coefficient...
## Intercept   -1.0438     0.2326  -4.488 0.000
##
## The intercept is the log(odds) a female will be unhealthy. This is because
## female is the first level in "sex" (the levels are ordered
## alphabetically by default, "F", "M")
female_log_odds = np.log(25 / 71)
print(female_log_odds)

## Now let's look at the second coefficient...
## sex[T.M]     1.2737     0.2725   4.674 0.000
##
## sex[T.M] is the log(odds ratio) that tells us that if a sample has sex=M, the
## odds of being unhealthy are, on a log scale, 1.27 times greater than if
## a sample has sex=F.
male_log_odds_ratio = np.log((112 / 89) / (25 / 71))
print(male_log_odds_ratio)

## Now calculate the overall "Pseudo R-squared" and its p-value

## NOTE: Since we are doing logistic regression...
## Null devaince = 2*(0 - LogLikelihood(null model))
##               = -2*LogLikihood(null model)
## Residual deviacne = 2*(0 - LogLikelihood(proposed model))
##                   = -2*LogLikelihood(proposed model)
ll_null = logistic.null_deviance / -2
ll_proposed = logistic.deviance / -2

## McFadden's Pseudo R^2 = [ LL(Null) - LL(Proposed) ] / LL(Null)
print((ll_null - ll_proposed) / ll_null)

## chi-square value = 2*(LL(Proposed) - LL(Null))
## p-value = 1 - chi2.cdf(chi-square value, df = 2-1)
print(1 - chi2.cdf(2 * (ll_proposed - ll_null), df=1))
print(1 - chi2.cdf(logistic.null_deviance - logistic.deviance, df=1))

## Lastly, let's  see what this logistic regression predicts, given
## that a patient is either female or male (and no other data about them).
predicted_df = pd.DataFrame(
    {"probability_of_hd": logistic.fittedvalues, "sex": df["sex"]}
)

## We can plot the data...
sex_colors = {"F": "#F8766D", "M": "#00BFC4"}

def draw_sex_plot():
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        x=predicted_df["sex"].map({"F": 1, "M": 2}),
        y=predicted_df["probability_of_hd"],
        c=predicted_df["sex"].map(sex_colors),
        s=150,
    )
    ax.set_xticks([1, 2], ["F", "M"])
    ax.set_xlim(0.5, 2.5)
    ax.set_xlabel("Sex")
    ax.set_ylabel("Predicted probability of getting heart disease")
    return fig


fig = draw_sex_plot()
fig.savefig("single_variable_logistic_py.pdf", transparent=True)
plt.show()

## Since there are only two probabilities (one for females and one for males),
## we can use a table to summarize the predicted probabilities.
print(pd.crosstab(predicted_df["probability_of_hd"], predicted_df["sex"]))

#####################################
##
## Now we will use all of the data available to predict heart disease
##
#####################################

## statsmodels needs a 0/1 outcome, so we model the probability that
## hd == "Unhealthy" (this is what R's glm() does, since "Unhealthy" is the
## second level of the factor)

## "hd ~ ." in R means "use all of the other columns". The formula interface
## in statsmodels has no "." shortcut, so we build the formula ourselves.
predictors = [col for col in df.columns if col not in ["hd"]]
# predictors = [col for col in df.columns if col not in ["hd", "unhealthy"]]
formula = "hd ~ " + " + ".join(predictors)
# formula = "unhealthy ~ " + " + ".join(predictors)

logistic = smf.glm(formula, 
                    data=df, 
                    family=sm.families.Binomial()).fit()
print(logistic.summary())

## Now calculate the overall "Pseudo R-squared" and its p-value
ll_null = logistic.null_deviance / -2
ll_proposed = logistic.deviance / -2

## McFadden's Pseudo R^2 = [ LL(Null) - LL(Proposed) ] / LL(Null)
print((ll_null - ll_proposed) / ll_null)

## The p-value for the R^2
print(1 - chi2.cdf(2 * (ll_proposed - ll_null), df=len(logistic.params) - 1))

print(logistic.fittedvalues.head(3))

## now we can plot the data
predicted_data = pd.DataFrame(
    {"probability_of_hd": logistic.fittedvalues, "hd": df["hd"]}
)

predicted_data = \
    predicted_data.sort_values("probability_of_hd", 
                                ascending=True)
predicted_data["rank"] = np.arange(1, 
                            len(predicted_data) + 1)

## Lastly, we can plot the predicted probabilities for each sample having
## heart disease and color by whether or not they actually had heart disease
## hd is coded 0 = Healthy, 1 = Unhealthy
hd_colors = {0: "#F8766D", 1: "#00BFC4"}
hd_labels = {0: "Healthy", 1: "Unhealthy"}

def draw_probability_plot():
    fig, ax = plt.subplots(figsize=(6, 6))
    for code, color in hd_colors.items():
        subset = predicted_data[predicted_data["hd"] == code]
        ax.scatter(
            subset["rank"],
            subset["probability_of_hd"],
            c=color,
            marker="x",
            linewidths=2,
            label=hd_labels[code],
        )
    ax.set_xlabel("Rank (low to high probability)")
    ax.set_ylabel("Predicted probability of getting heart disease")
    ax.legend(loc="upper left", title="hd")
    return fig


fig = draw_probability_plot()
fig.savefig("full_model_logistic_py.pdf", transparent=True)
plt.show()

#####################################
##
## Now let's use the full model to predict the probability that new
## (made up) patients have heart disease
##
#####################################

## NOTE: The new data must have the same columns as the data used to fit the
## model (except "hd", which is what we are predicting), and the categorical
## columns must use the same levels. Setting categories = df[col].cat.categories
## takes care of that.
new_patients = pd.DataFrame(
    {
        "age": [58, 35],
        "sex": ["M", "F"],
        "cp": ["4", "3"],
        "trestbps": [140, 115],
        "chol": [250, 190],
        "fbs": ["0", "0"],
        "thalach": [130, 175],
        "exang": ["1", "0"],
        "oldpeak": [1.5, 0],
        "slope": ["2", "1"],
        "ca": ["1", "0"],
        "thal": ["7", "3"],
    }
)
for col in ["sex", 
            "cp", 
            "fbs", 
            "exang", 
            "slope", 
            "ca", 
            "thal"]:
    new_patients[col] = pd.Categorical(
        new_patients[col], categories=df[col].cat.categories
    )

## predict() on a GLM returns probabilities (not log(odds))
new_patients["probability_of_hd"] = logistic.predict(new_patients)
print(new_patients[["age", "sex", "cp", "probability_of_hd"]])

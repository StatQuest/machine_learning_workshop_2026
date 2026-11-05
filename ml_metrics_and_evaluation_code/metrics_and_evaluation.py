import matplotlib.pyplot as plt  # plotting
import numpy as np  # numerical arrays and random number generation
import pandas as pd  # data frames
import statsmodels.api as sm  # logistic regression (equivalent of R's glm)
from scipy.stats import rankdata  # equivalent of R's rank()
from sklearn.ensemble import RandomForestClassifier  # equivalent of randomForest
from sklearn.metrics import (  # ROC, AUC, confusion matrix
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

#######################################
##
## Generate blood pressure and heart disease datasets.
##
#######################################
# NOTE: numpy's random number generator differs from R's, so the values
# will not match the R output even with the same seed
rng = np.random.default_rng(seed=42)  # create a seeded random number generator

num_samples = 100  # number of samples to generate

# generate 100 values from a normal distribution with mean 172 and sd 29, then sort them
bp = np.sort(rng.normal(loc=172, scale=29, size=num_samples))

# rank(bp) is 1 for the lowest blood pressure ... 100 for the highest.
# A sample has heart disease (1) if a uniform random number is < rank/100
hd = np.where(
    rng.uniform(size=num_samples) < (rankdata(bp) / num_samples),  # compare random number to rank/100
    1,  # yes: has heart disease
    0,  # no: does not have heart disease
)
print(hd)  # show which samples have heart disease (1) and which do not (0)

#######################################
##
## Now fit a logistic regression to the data
## and draw the data with the fitted squiggle
##
#######################################

# add an intercept column to bp, since statsmodels does not add one automatically
X = sm.add_constant(bp)
X[0:3,] # print the first few rows
# fit a logistic regression: hd ~ bp, with a binomial family
glm_fit = sm.GLM(
    hd, 
    X, 
    family=sm.families.Binomial()
    ).fit()

# extract the fitted probabilities (R: glm.fit$fitted.values)
fitted_values = glm_fit.fittedvalues
fitted_values[0:3] # to verify

# create a new figure for the data plot
plt.figure()
# plot the data, coloring points by heart disease status
plt.scatter(
    bp,  # x-axis: blood pressure
    hd,  # y-axis: heart disease (0/1)
    c=np.where(hd == 1, "#1b9e77", "#7570b3"),  # color by class
)
# draw the logistic regression squiggle
plt.plot(
    bp, 
    fitted_values, 
    linewidth=3, 
    color="#d95f02"
    )
plt.xlabel("Blood Pressure")  # x-axis label
plt.ylabel("Heart Disease")  # y-axis label
plt.show()  # display the plot

plt.savefig("logistic_regression_python.pdf", transparent=True)  # save with a transparent background

#######################################
##
## draw confusion matrix and calculate
## sensitivity, specificity and precision
##
#######################################



# predict heart disease (1) when the fitted probability is > 0.5
logistic_predicted = (fitted_values > 0.5).astype(int)
# build the confusion matrix; rows = Predicted, columns = Actual
# (sklearn puts Actual on rows, so we transpose to match R's table layout)
logistic_conf_matrix = pd.DataFrame(
    confusion_matrix(
        hd, 
        logistic_predicted, 
        labels=[0, 1]
        ).T,  # transpose: Predicted x Actual
    index=pd.Index([0, 1], name="Predicted"),  # row labels
    columns=pd.Index([0, 1], name="Actual"),  # column labels
)
# display the confusion matrix
print(logistic_conf_matrix)

# now calculate metrics
# predicted 1, actual 1
logistic_TP = logistic_conf_matrix.loc[1, 1]  
print(logistic_TP)  

# predicted 0, actual 0
logistic_TN = logistic_conf_matrix.loc[0, 0]  
print(logistic_TN)  

# predicted 1, actual 0
logistic_FP = logistic_conf_matrix.loc[1, 0]  
print(logistic_FP)  

# predicted 0, actual 1
logistic_FN = logistic_conf_matrix.loc[0, 1]  
print(logistic_FN)  

# accuracy = correct predictions / total samples
logistic_accuracy = \
    (logistic_TP + logistic_TN) / len(hd)
print(logistic_accuracy)  # show accuracy

# sensitivity (recall) = TP / (TP + FN)
logistic_sensitivity = \
    logistic_TP / (logistic_TP + logistic_FN)
print(logistic_sensitivity)  # show sensitivity

# specificity = TN / (TN + FP)
logistic_specificity = \
    logistic_TN / (logistic_TN + logistic_FP)
print(logistic_specificity)  # show specificity

# precision = TP / (TP + FP)
logistic_precision = \
    logistic_TP / (logistic_TP + logistic_FP)
print(logistic_precision)  # show precision

#######################################
##
## draw ROC and AUC for the logistic regression
##
#######################################

# compute the ROC curve: false positive rate, true positive rate, thresholds
# (drop_intermediate=False keeps every threshold, like pROC)
fpr, tpr, thresholds = roc_curve(
    hd, 
    fitted_values, 
    drop_intermediate=False
    )
# compute the area under the curve, as a percentage to match percent = TRUE
logistic_auc = roc_auc_score(
    hd, 
    fitted_values
    ) * 100

# create a square figure (R: par(pty = "s"))
fig, ax = plt.subplots(figsize=(6, 6))
# draw the ROC curve in percent units
ax.plot(
    fpr * 100, 
    tpr * 100, 
    color="#377eb8", 
    linewidth=4)
ax.set_xlabel("False Positive Percentage")  # x-axis label
ax.set_ylabel("True Postive Percentage")  # y-axis label
ax.set_aspect("equal")  # force a square plotting region
ax.text(50, 50, f"AUC: {logistic_auc:.1f}%")  # print the AUC on the plot
plt.show()  # display the plot
# save a 4x4 inch version of the ROC plot
# fig.set_size_inches(4, 4)  # resize the figure
# fig.savefig("logstic_roc_python.pdf", transparent=True)  # write the PDF



# store the data used to make the ROC graph in a data frame
roc_df = pd.DataFrame(
    {
        "tpp": tpr * 100,  # true positive percentage
        "fpp": fpr * 100,  # false positive percentage
        "thresholds": thresholds,  # thresholds (first is inf: everything predicted positive is none)
    }
)

# look at the thresholds with TPP between 70% and 80%
print(
    roc_df[(roc_df["tpp"] > 70) & (roc_df["tpp"] < 80)])

#######################################
##
## Now let's fit the data with a random forest...
##
#######################################
# fit a random forest of 500 trees (R default) predicting hd from bp;
# oob_score=True lets us get out-of-bag votes like R's rf.model$votes
rf_model = RandomForestClassifier(
    n_estimators=500, 
    oob_score=True, 
    random_state=42)
rf_model.fit(bp.reshape(-1, 1), hd)  # sklearn expects a 2D feature array
# out-of-bag fraction of votes for class 1
# (R used votes[, 1], the class-0 votes; pROC then auto-flips the direction,
# so using class-1 votes gives the same ROC curve and AUC)
rf_votes = rf_model.oob_decision_function_[:, 1]

###########################
##
## Now draw the ROC
##
############################

# ROC curve for the random forest
rf_fpr, rf_tpr, _ = roc_curve(hd, rf_votes)  # false/true positive rates
rf_auc = roc_auc_score(hd, rf_votes) * 100  # AUC as a percentage

#######################################
##
## Now layer logistic regression and random forest ROC graphs..
##
#######################################
# create a square figure to hold both ROC curves
fig, ax = plt.subplots(figsize=(6, 6))
# logistic regression ROC
ax.plot(
    fpr * 100, 
    tpr * 100, 
    color="#377eb8", 
    linewidth=4, 
    label="Logisitic Regression"
    )
# random forest ROC layered on top
ax.plot(
    rf_fpr * 100, 
    rf_tpr * 100, 
    color="#4daf4a", 
    linewidth=4, 
    label="Random Forest"
    )
ax.set_xlabel("False Positive Percentage")  # x-axis label
ax.set_ylabel("True Postive Percentage")  # y-axis label
ax.set_aspect("equal")  # square plotting region
ax.text(
    50, 
    50, 
    f"AUC: {logistic_auc:.1f}%", 
    color="#377eb8"
    )  # logistic AUC
ax.text(
    50,
    40, 
    f"AUC: {rf_auc:.1f}%", 
    color="#4daf4a"
    )  # random forest AUC
ax.legend(loc="lower right")  # legend in the bottom right
plt.show()  # display the plot

# save the layered plot as a 6x6 inch PDF
fig.set_size_inches(6, 6)  # resize the figure
fig.savefig("logistic_vs_forest_AUC_python.pdf", transparent=True)  # write the PDF

#######################################
##
## Reset plot settings. (R: par(pty = "m"))
## matplotlib needs no reset because aspect ratio was set per-axes.
##
#######################################
plt.rcdefaults()  # restore default matplotlib settings

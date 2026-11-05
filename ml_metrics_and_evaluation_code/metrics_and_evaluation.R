library(pROC) # install with install.packages("pROC")
library(randomForest) # install with install.packages("randomForest")

#######################################
##
## Generate blood pressure and heart disease datasets.
##
#######################################
set.seed(42) # this will make my results match yours

num.samples <- 100

## genereate 100 values from a normal distribution with
## mean 172 and standard deviation 29, then sort them
bp <- sort(rnorm(n = num.samples, mean = 172, sd = 29))

## Now we will decide if a sample has heart disease or not.
## NOTE: This method for classifying a sample as having heart disease or not
## was made up just for this example.
## rank(bp) returns 1 for the lowest blood pressure, 2 for the second lowest, ...
##              ... and it returns 100 for the highest blood pressure
## So what we do is generate a random number between 0 and 1. Then we see if
## that number is less than rank/100. So, for the lowest sample, rank = 1.
## This sample will be classified having heart disease if we get a random number less than
## 1/100. For the second lowest sample, rank = 2, we get another random
## number between 0 and 1 and classify this sample having heart disease if that random
## number is < 2/100. We repeat that process for all 100 samples
hd <- ifelse(
  test = (runif(n = num.samples) < (rank(bp) / num.samples)),
  yes = 1,
  no = 0
)
hd ## print out the contents of "hd" to show us which samples were
## classified as having heart disease with 1, and which samples were classified
## "not having heart disease" with 0.

#######################################
##
## Now fit a logistic regression to the data
## and draw the data with the fitted squiggle
##
#######################################

## fit a logistic regression to the data...
glm.fit <- glm(hd ~ bp, family = binomial)

## plot the data
plot(
  x = bp,
  y = hd,
  col = ifelse(hd == 1, "#1b9e77", "#7570b3"),
  pch = 19,
  xlab = "Blood Pressure",
  ylab = "Heart Disease"
)

## plot the logistic regression squiggle
lines(bp, glm.fit$fitted.values, lwd = 3, col = "#d95f02")

# pdf("logistic_regression.pdf", width = 6, height = 4)
# plot(
#   x = bp,
#   y = hd,
#   col = ifelse(hd == 1, "#1b9e77", "#7570b3"),
#   pch = 19,
#   cex = 2,
#   xlab = "Blood Pressure",
#   ylab = "Heart Disease"
# )
# lines(bp, glm.fit$fitted.values, lwd = 3, col = "#d95f02")
# dev.off()

#######################################
##
## draw confusion matrix and calculate
## sensitivity, specificity and precision
##
#######################################

logistic.predicted <- factor(
  glm.fit$fitted.values > 0.5,
  levels = c(FALSE, TRUE),
  labels = c(0, 1)
)
logistic.conf.matrix <- table(
  Predicted = logistic.predicted,
  Actual = factor(hd)
)
# display the confusion matrix
logistic.conf.matrix

# now calculate metrics
logistic.TP <- logistic.conf.matrix["1", "1"]
logistic.TP
logistic.TN <- logistic.conf.matrix["0", "0"]
logistic.TN
logistic.FP <- logistic.conf.matrix["1", "0"]
logistic.FP
logistic.FN <- logistic.conf.matrix["0", "1"]
logistic.FN

logistic.accuracy <- (logistic.TP + logistic.TN) / length(hd)
logistic.accuracy

logistic.sensitivity <-
  logistic.TP / (logistic.TP + logistic.FN)
logistic.sensitivity

logistic.specificity <-
  logistic.TN / (logistic.TN + logistic.FP)
logistic.specificity

logistic.precision <-
  logistic.TP / (logistic.TP + logistic.FP)
logistic.precision

#######################################
##
## draw ROC and AUC for the logistic regression using pROC
##
#######################################

## NOTE: By default, the graphs come out looking terrible
## The problem is that ROC graphs should be square, since the x and y axes
## both go from 0 to 1. However, the window in which I draw them isn't square
## so extra whitespace is added to pad the sides.
#roc(hd, glm.fit$fitted.values, plot = TRUE)

## Now let's configure R so that it prints the graph as a square.
##
par(pty = "s") ## pty sets the aspect ratio of the plot region. Two options:
##                "s" - creates a square plotting region
##                "m" - (the default) creates a maximal plotting region
#roc(hd, glm.fit$fitted.values, plot = TRUE)

## NOTE: By default, roc() uses specificity on the x-axis and the values range
## from 1 to 0. This makes the graph look like what we would expect, but the
## x-axis itself might induce a headache. To use 1-specificity (i.e. the
## False Positive Rate) on the x-axis, set "legacy.axes" to TRUE.
#roc(hd, glm.fit$fitted.values, plot = TRUE, legacy.axes = TRUE)

## If you want to rename the x and y axes...
# roc(
#   hd,
#   glm.fit$fitted.values,
#   plot = TRUE,
#   legacy.axes = TRUE,
#   percent = TRUE,
#   xlab = "False Positive Percentage",
#   ylab = "True Postive Percentage"
# )

## We can also change the color of the ROC line, and make it wider...
roc(
  hd,
  glm.fit$fitted.values,
  plot = TRUE,
  legacy.axes = TRUE,
  percent = TRUE,
  xlab = "False Positive Percentage",
  ylab = "True Postive Percentage",
  col = "#377eb8",
  lwd = 4
)

pdf("logstic_roc.pdf", width = 4, height = 4)
roc(
  hd,
  glm.fit$fitted.values,
  plot = TRUE,
  legacy.axes = TRUE,
  percent = TRUE,
  xlab = "False Positive Percentage",
  ylab = "True Postive Percentage",
  col = "#377eb8",
  lwd = 4
)
dev.off()
## If we want to find out the optimal threshold we can store the
## data used to make the ROC graph in a variable...
roc.info <- roc(hd, glm.fit$fitted.values, legacy.axes = TRUE)
# str(roc.info)

## and then extract just the information that we want from that variable.
roc.df <- data.frame(
  tpp = roc.info$sensitivities * 100, ## tpp = true positive percentage
  fpp = (1 - roc.info$specificities) * 100, ## fpp = false positive precentage
  thresholds = roc.info$thresholds
)

head(roc.df, n = 3) ## head() will show us the values for the upper right-hand corner
## of the ROC graph, when the threshold is so low
## (negative infinity) that every single sample is predicted
## as having heart disease
## Thus TPP = 100% and FPP = 100%

# tail(roc.df) ## tail() will show us the values for the lower left-hand corner
## of the ROC graph, when the threshold is so high (infinity)
## that every single sample is predicted to not have heart disease.
## Thus, TPP = 0% and FPP = 0%

## now let's look at the thresholds between TPP 70% and 80%...
roc.df[roc.df$tpp > 70 & roc.df$tpp < 80, ]

## We can calculate the area under the curve...
# roc(
#   hd,
#   glm.fit$fitted.values,
#   plot = TRUE,
#   legacy.axes = TRUE,
#   percent = TRUE,
#   xlab = "False Positive Percentage",
#   ylab = "True Postive Percentage",
#   col = "#377eb8",
#   lwd = 4,
#   print.auc = TRUE
# )

# ## ...and the partial area under the curve.
# roc(
#   hd,
#   glm.fit$fitted.values,
#   plot = TRUE,
#   legacy.axes = TRUE,
#   percent = TRUE,
#   xlab = "False Positive Percentage",
#   ylab = "True Postive Percentage",
#   col = "#377eb8",
#   lwd = 4,
#   print.auc = TRUE,
#   print.auc.x = 45,
#   partial.auc = c(100, 90),
#   auc.polygon = TRUE,
#   auc.polygon.col = "#377eb822"
# )

#######################################
##
## Now let's fit the data with a random forest...
##
#######################################
rf.model <- randomForest(factor(hd) ~ bp)


###########################
##
## Now draw the ROC
##
############################

## ROC for random forest
roc(
  hd,
  rf.model$votes[, 1],
  plot = TRUE,
  legacy.axes = TRUE,
  percent = TRUE,
  xlab = "False Positive Percentage",
  ylab = "True Postive Percentage",
  col = "#4daf4a",
  lwd = 4,
  print.auc = TRUE
)


#######################################
##
## Now layer logistic regression and random forest ROC graphs..
##
#######################################
roc(
  hd,
  glm.fit$fitted.values,
  plot = TRUE,
  legacy.axes = TRUE,
  percent = TRUE,
  xlab = "False Positive Percentage",
  ylab = "True Postive Percentage",
  col = "#377eb8",
  lwd = 4,
  print.auc = TRUE
)

plot.roc(
  hd,
  rf.model$votes[, 1],
  percent = TRUE,
  col = "#4daf4a",
  lwd = 4,
  print.auc = TRUE,
  add = TRUE,
  print.auc.y = 40
)
legend(
  "bottomright",
  legend = c("Logisitic Regression", "Random Forest"),
  col = c("#377eb8", "#4daf4a"),
  lwd = 4
)

pdf("logistic_vs_forest_AUC.pdf", width = 6, height = 6)
roc(
  hd,
  glm.fit$fitted.values,
  plot = TRUE,
  legacy.axes = TRUE,
  percent = TRUE,
  xlab = "False Positive Percentage",
  ylab = "True Postive Percentage",
  col = "#377eb8",
  lwd = 4,
  print.auc = TRUE
)

plot.roc(
  hd,
  rf.model$votes[, 1],
  percent = TRUE,
  col = "#4daf4a",
  lwd = 4,
  print.auc = TRUE,
  add = TRUE,
  print.auc.y = 40
)
legend(
  "bottomright",
  legend = c("Logisitic Regression", "Random Forest"),
  col = c("#377eb8", "#4daf4a"),
  lwd = 4
)
dev.off()

#######################################
##
## Now that we're done with our ROC fun, let's reset the par() variables.
## There are two ways to do it...
##
#######################################
par(pty = "m")

## NOTE: The data used in this demo comes from the UCI machine learning
## repository.
## http://archive.ics.uci.edu/ml/index.php
## Specifically, this is the heart disease data set.
## http://archive.ics.uci.edu/ml/datasets/Heart+Disease

df <- read.csv("processed_cleveland_data.csv", header = FALSE)

#####################################
##
## Reformat the df so that it is
## 1) Easy to use (add nice column names)
## 2) Interpreted correctly by glm()..
##
#####################################
head(df) # you see data, but no column names

colnames(df) <- c(
  "age",
  "sex", # 0 = female, 1 = male
  "cp", # chest pain
  # 1 = typical angina,
  # 2 = atypical angina,
  # 3 = non-anginal pain,
  # 4 = asymptomatic
  "trestbps", # resting blood pressure (in mm Hg)
  "chol", # serum cholestoral in mg/dl
  "fbs", # fasting blood sugar if less than 120 mg/dl, 1 = TRUE, 0 = FALSE
  "restecg", # resting electrocardiographic results
  # 1 = normal
  # 2 = having ST-T wave abnormality
  # 3 = showing probable or definite left ventricular hypertrophy
  "thalach", # maximum heart rate achieved
  "exang", # exercise induced angina, 1 = yes, 0 = no
  "oldpeak", # ST depression induced by exercise relative to rest
  "slope", # the slope of the peak exercise ST segment
  # 1 = upsloping
  # 2 = flat
  # 3 = downsloping
  "ca", # number of major vessels (0-3) colored by fluoroscopy
  "thal", # this is short of thalium heart scan
  # 3 = normal (no cold spots)
  # 6 = fixed defect (cold spots during rest and exercise)
  # 7 = reversible defect (when cold spots only appear during exercise)
  "hd" # (the predicted attribute) - diagnosis of heart disease
  # 0 if less than or equal to 50% diameter narrowing
  # 1 if greater than 50% diameter narrowing
)

head(df, n = 3) # now we have data and column names

str(df) # this shows that we need to tell R which columns contain factors
# it also shows us that there are some missing values. There are "?"s
# in the dataset. These are in the "ca" and "thal" columns...
unique(df$ca)

unique(df$thal)

## First, convert "?"s to NAs...
df[df == "?"] <- NA

df$ca <- as.integer(df$ca) # since this column had "?"s in it
# R thinks that the levels for the factor are strings, but
# we know they are integers, so first convert the strings to integers...

df$thal <- as.integer(df$thal) # "thal" also had "?"s in it.

## Now add factors for variables that are factors and clean up the factors
## that had missing data...
df[df$sex == 0, ]$sex <- "F"
df[df$sex == 1, ]$sex <- "M"
df$sex <- as.factor(df$sex)

df$cp <- as.factor(df$cp)
df$fbs <- as.factor(df$fbs)
df$restecg <- as.factor(df$restecg)
df$exang <- as.factor(df$exang)
df$slope <- as.factor(df$slope)

df$ca <- as.factor(df$ca) # ...then convert the integers to factor levels
df$thal <- as.factor(df$thal)


## This next line replaces 0 and 1 with "Healthy" and "Unhealthy"
unique(df$hd)
df$hd <-
  ifelse(test = df$hd == 0, yes = "Healthy", no = "Unhealthy")
df$hd <- as.factor(df$hd) # Now convert to a factor

str(df) ## this shows that the correct columns are factors

## Now determine how many rows have "NA" (aka "Missing data"). If it's just
## a few, we can remove them from the dataset, otherwise we should consider
## imputing the values with a Random Forest or some other imputation method.
nrow(df[is.na(df$ca) | is.na(df$thal), ])
df[is.na(df$ca) | is.na(df$thal), ]
nrow(df)

nrow(df[is.na(df$ca) | is.na(df$thal), ]) / nrow(df)
## so 6 of the 303 rows of data have missing values. This isn't a large
## percentage (2%), so we can just remove them from the dataset
## NOTE: This is different from when we did machine learning with
## Random Forests. When we did that, we imputed values.
nrow(df)
df <- df[!(is.na(df$ca) | is.na(df$thal)), ]
nrow(df)

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
xtabs(~ hd + sex, data = df)
xtabs(~ hd + cp, data = df)
xtabs(~ hd + fbs, data = df)
xtabs(~ hd + restecg, data = df)
df <- subset(df, select = -restecg)
head(df)
xtabs(~ hd + exang, data = df)
xtabs(~ hd + slope, data = df)
xtabs(~ hd + ca, data = df)
xtabs(~ hd + thal, data = df)

#####################################
##
## Now we are ready for some logistic regression. First we'll create a very
## simple model that uses sex to predict heart disease
##
#####################################

## let's start super simple and see if sex (female/male) is a good
## predictor...
## First, let's just look at the raw data...
xtabs(~ hd + sex, data = df)
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

logistic <- glm(hd ~ sex, data = df, family = "binomial")
summary(logistic)
## (Intercept)  -1.0438     0.2326  -4.488 7.18e-06 ***
##   sexM        1.2737     0.2725   4.674 2.95e-06 ***

## Let's start by going through the first coefficient...
## (Intercept)  -1.0438     0.2326  -4.488 7.18e-06 ***
##
## The intercept is the log(odds) a female will be unhealthy. This is because
## female is the first factor in "sex" (the factors are ordered,
## alphabetically by default,"female", "male")
female.log.odds <- log(25 / 71)
female.log.odds

## Now let's look at the second coefficient...
##   sexM        1.2737     0.2725   4.674 2.95e-06 ***
##
## sexM is the log(odds ratio) that tells us that if a sample has sex=M, the
## odds of being unhealthy are, on a log scale, 1.27 times greater than if
## a sample has sex=F.
male.log.odds.ratio <- log((112 / 89) / (25 / 71))
male.log.odds.ratio

## Now calculate the overall "Pseudo R-squared" and its p-value

## NOTE: Since we are doing logistic regression...
## Null devaince = 2*(0 - LogLikelihood(null model))
##               = -2*LogLikihood(null model)
## Residual deviacne = 2*(0 - LogLikelihood(proposed model))
##                   = -2*LogLikelihood(proposed model)
ll.null <- logistic$null.deviance / -2
ll.proposed <- logistic$deviance / -2

## McFadden's Pseudo R^2 = [ LL(Null) - LL(Proposed) ] / LL(Null)
(ll.null - ll.proposed) / ll.null

## chi-square value = 2*(LL(Proposed) - LL(Null))
## p-value = 1 - pchisq(chi-square value, df = 2-1)
1 - pchisq(2 * (ll.proposed - ll.null), df = 1)
1 - pchisq((logistic$null.deviance - logistic$deviance), df = 1)

## Lastly, let's  see what this logistic regression predicts, given
## that a patient is either female or male (and no other data about them).
predicted.df <- data.frame(
  probability.of.hd = logistic$fitted.values,
  sex = df$sex
)

## We can plot the data...
sex.colors <- c(F = "#F8766D", M = "#00BFC4")
plot(
  x = as.integer(predicted.df$sex),
  y = predicted.df$probability.of.hd,
  col = sex.colors[as.character(predicted.df$sex)],
  pch = 16,
  cex = 3,
  xaxt = "n",
  xlim = c(0.5, 2.5),
  xlab = "Sex",
  ylab = "Predicted probability of getting heart disease"
)
axis(side = 1, at = 1:2, labels = levels(predicted.df$sex))

pdf(
  file = "single_variable_logistic.pdf",
  width = 6,
  height = 6,
  bg = "transparent"
)
plot(
  x = as.integer(predicted.df$sex),
  y = predicted.df$probability.of.hd,
  col = sex.colors[as.character(predicted.df$sex)],
  pch = 16,
  cex = 3,
  xaxt = "n",
  xlim = c(0.5, 2.5),
  xlab = "Sex",
  ylab = "Predicted probability of getting heart disease"
)
axis(side = 1, at = 1:2, labels = levels(predicted.df$sex))
dev.off()

## Since there are only two probabilities (one for females and one for males),
## we can use a table to summarize the predicted probabilities.
xtabs(~ probability.of.hd + sex, data = predicted.df)

#####################################
##
## Now we will use all of the data available to predict heart disease
##
#####################################

logistic <- glm(hd ~ ., data = df, family = "binomial")
summary(logistic)

## Now calculate the overall "Pseudo R-squared" and its p-value
ll.null <- logistic$null.deviance / -2
ll.proposed <- logistic$deviance / -2

## McFadden's Pseudo R^2 = [ LL(Null) - LL(Proposed) ] / LL(Null)
(ll.null - ll.proposed) / ll.null

## The p-value for the R^2
1 -
  pchisq(2 * (ll.proposed - ll.null), df = (length(logistic$coefficients) - 1))

head(logistic$fitted.values, n = 3)

## now we can plot the data
predicted.data <- data.frame(
  probability.of.hd = logistic$fitted.values,
  hd = df$hd
)

predicted.data <- predicted.data[
  order(predicted.data$probability.of.hd, decreasing = FALSE),
]
predicted.data$rank <- 1:nrow(predicted.data)

## Lastly, we can plot the predicted probabilities for each sample having
## heart disease and color by whether or not they actually had heart disease
hd.colors <- c(Healthy = "#F8766D", Unhealthy = "#00BFC4")
plot(
  x = predicted.data$rank,
  y = predicted.data$probability.of.hd,
  col = hd.colors[as.character(predicted.data$hd)],
  pch = 4,
  lwd = 2,
  xlab = "Rank (low to high probability)",
  ylab = "Predicted probability of getting heart disease"
)
legend(
  "topleft",
  legend = names(hd.colors),
  col = hd.colors,
  pch = 4,
  pt.lwd = 2,
  title = "hd"
)

pdf(file = "full_model_logistic.pdf", width = 6, height = 6, bg = "transparent")
plot(
  x = predicted.data$rank,
  y = predicted.data$probability.of.hd,
  col = hd.colors[as.character(predicted.data$hd)],
  pch = 4,
  lwd = 2,
  xlab = "Rank (low to high probability)",
  ylab = "Predicted probability of getting heart disease"
)
legend(
  "topleft",
  legend = names(hd.colors),
  col = hd.colors,
  pch = 4,
  pt.lwd = 2,
  title = "hd"
)
dev.off()

#####################################
##
## Now let's use the full model to predict the probability that new
## (made up) patients have heart disease
##
#####################################

## NOTE: The new data must have the same columns as the data used to fit the
## model (except "hd", which is what we are predicting), and the factor columns
## must use the same levels. Setting levels = levels(df$...) takes care of that.
new.patients <- data.frame(
  age = c(58, 35),
  sex = factor(c("M", "F"), levels = levels(df$sex)),
  cp = factor(c("4", "3"), levels = levels(df$cp)),
  trestbps = c(140, 115),
  chol = c(250, 190),
  fbs = factor(c("0", "0"), levels = levels(df$fbs)),
  thalach = c(130, 175),
  exang = factor(c("1", "0"), levels = levels(df$exang)),
  oldpeak = c(1.5, 0),
  slope = factor(c("2", "1"), levels = levels(df$slope)),
  ca = factor(c("1", "0"), levels = levels(df$ca)),
  thal = factor(c("7", "3"), levels = levels(df$thal))
)

## type = "response" returns probabilities instead of log(odds)
new.patients$probability.of.hd <- predict(
  logistic,
  newdata = new.patients,
  type = "response"
)
new.patients[, c("age", "sex", "cp", "probability.of.hd")]

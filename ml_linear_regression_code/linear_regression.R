# first read the data into a data frame
df <- read.csv("Salary_dataset.csv")

# look at the first few rows
head(df)

# the first colun X, is just the row
# number, so we can omit it.
df <- subset(df, select = -X)

# confirm that we removed X
head(df)

# now plot the data
plot(
  df$YearsExperience,
  df$Salary,
  xlab = "Years of Experience",
  ylab = "Salary"
)

# save a PDF of the graph with a transparent
# background
pdf("salary_data_plot.pdf", width = 4, height = 4, bg = "transparent")
plot(
  df$YearsExperience,
  df$Salary,
  xlab = "Years of Experience",
  ylab = "Salary"
)
dev.off()

# now do linear regression with the data
model <- lm(Salary ~ YearsExperience, data = df)
summary(model)

# now add the regression line to our graph
plot(df$YearsExperience, df$Salary)
abline(model, col = "red", lwd = 2)

# save a PDF of the graph with a transparent
# background
pdf("linear_regression.pdf", width = 4, height = 4, bg = "transparent")
plot(
  df$YearsExperience,
  df$Salary,
  xlab = "Years of Experience",
  ylab = "Salary"
)
abline(model, col = "red", lwd = 2)
dev.off()

# now summarize the range of YearsExperience
# that are reasonable input values for
# our model
summary(df$YearsExperience)

# now predict the salary for someone
# with 6.3 years of experience
predict(model, newdata = data.frame(YearsExperience = 6.3))

# Bam!

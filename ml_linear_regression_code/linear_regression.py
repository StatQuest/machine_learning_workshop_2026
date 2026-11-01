import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

# first read the data into a data frame
df = pd.read_csv("Salary_dataset.csv")

# look at the first few rows
print(df.head())

# the first column, "Unnamed: 0",
# is just the row number, so we can omit it.
df = df.drop(columns=df.columns[0])

# confirm that we removed the correct column
print(df.head())

# now plot the data
fig, ax = plt.subplots(figsize=(4, 4))
sns.scatterplot(data=df, 
    x="YearsExperience", 
    y="Salary", 
    ax=ax)
ax.set_xlabel("Years of Experience")
ax.set_ylabel("Salary")
plt.show()

# save a PDF of the graph with a transparent
# background
fig.savefig("salary_data_plot_python.pdf", transparent=True, bbox_inches="tight")

# now do linear regression with the data
model = smf.ols("Salary ~ YearsExperience", data=df).fit()
print(model.summary())

# now add the regression line to our graph
fig, ax = plt.subplots(figsize=(4, 4))
sns.scatterplot(
    data=df, 
    x="YearsExperience", 
    y="Salary", 
    ax=ax
)
sns.regplot(
    data=df,
    x="YearsExperience",
    y="Salary",
    scatter=False,
    ci=None,
    color="red",
    line_kws={"linewidth": 2},
    ax=ax,
)
ax.set_xlabel("Years of Experience")
ax.set_ylabel("Salary")
plt.show()

# save a PDF of the graph with a transparent
# background
fig.savefig("linear_regression_python.pdf", transparent=True, bbox_inches="tight")

# now summarize the range of YearsExperience
# that are reasonable input values for
# our model
print(df["YearsExperience"].describe())


# now predict the salary for someone
# with 6.3 years of experience
new_data = pd.DataFrame({"YearsExperience": [6.3]})
print(model.predict(new_data))

# Bam!

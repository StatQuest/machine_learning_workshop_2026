import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

path = "~/My Drive/stat_quests/course_intro_to_ml_2026/00_guides/code_and_stuff/"

pandas_df = pd.read_csv(path + "example_data.txt", sep="\t")
# pandas_df = pd.read_csv(path + "example_data.txt", sep="\t", header=None)

# print out the first 5 rows
pandas_df.head()
# print out the first 3 rows
pandas_df.head(n=3)


# demonstrate how to add column names if we needed to...
pandas_df.columns = ["ID", 
                        "age", 
                        "height", 
                        "col", 
                        "name", 
                        "date", 
                        "soda"]

# print out the first 3 rows
pandas_df.head(n=3)

# demonstrate how to rename specific columns
pandas_df.rename({
    'ID' : 'uniqueID',
    'col' : 'color'}, 
    axis='columns', 
    inplace=True)

# print out the first 3 rows
pandas_df.head(n=3)


# Drop uniformitive columns
pandas_df.drop(['uniqueID', 'name'], 
                axis='columns', 
                inplace=True)
pandas_df.head(n=3)

# to check for missing or messed up data
# one of the first things we can do is make sure
# each column in the dataframe is the expected type
# in other words, columns that have numbers in them
# should probably be of some numeric type and columns
# with text in them (like "colors"), should be of some
# string type. We can check the data type for each
# column with the .dtypes variable.
print(pandas_df.dtypes)

# The float64 value for "age" tells us something is
# wrong in that column because it should just be integer
# values. In this case, the problem is the "NaN" in the 
# first row, which is defined as a floating point number,
# so everything that column is treated as such. 
# Note: When we can't easily see an NaN value, we can search for
# them by counting how many are in a column like this.
pandas_df["age"].isna().sum()
# Now convert the missing value "NaN" to pd.NA
# Note: We can't use inplace=True here since pandas 3.0 is dropping
# that feature. 
pandas_df["age"] = pandas_df["age"].replace([float("nan")], pd.NA)
# verify that NaN is now NA
pandas_df.head(n=3)
# verify that data type is now integer
print(pandas_df.dtypes)
# since it's not, we can convert the dtype for the age column to int64
pandas_df["age"] = pandas_df["age"].astype("Int64")
# now verify the data type
print(pandas_df.dtypes)

# now let's convert the missing value ??? in the height column to pd.NA
pandas_df["height"] = pandas_df["height"].replace("???", pd.NA)
pandas_df.head(n=3)
print(pandas_df.dtypes)
# now let's convert the dtype for the height column to Float64
pandas_df["height"] = pandas_df["height"].astype("Float64")
print(pandas_df.dtypes)


# The the "color" column has all kinds of
# capitalization and we can see this by 
# printing out the unique values...
print(pandas_df["color"].unique())
# and convert all of them to lowercase
pandas_df["color"] = pandas_df["color"].str.lower()
print(pandas_df["color"].unique()) # verify
pandas_df.head(n=3)

# Now we need to fix the date values in the "date" column
# Step 1: remove any whitespace...
pandas_df["date"] = pandas_df["date"].str.strip()

# Step 2: parse each value on its own, since the formats are mixed (assumes month-first)
pandas_df["date"] = pd.to_datetime(pandas_df["date"], 
                                    format="mixed", 
                                    errors="coerce")
pandas_df.head(n=3)
print(pandas_df.dtypes)

# Step 3: two-digit years like '73 get read as 2073, so any date in the future
# is really from the 1900s. Subtract 100 years from those dates
future = pandas_df["date"] > pd.Timestamp.today()

# inspect some future dates to see the problem
pandas_df.loc[future].head(n=3)

# fix the problem
pandas_df.loc[future, "date"] -= pd.DateOffset(years=100)

# inspect some future dates to see if they were corrected
pandas_df.loc[future].head(n=3)


# lastly, let's look at the different values in the "soda" column
print(pandas_df["soda"].unique())
# and we see all kind of different capitalization,
# so let's make all the values in the "soda" lowercase
pandas_df["soda"] = pandas_df["soda"].str.lower()
# verify
print(pandas_df["soda"].unique())
# and just look at the values
pandas_df.head(n=3)

# Before we fix the dates, let's just look at them
print(pandas_df)




# Now, just to show the method, we'll downsample the data...
print(len(pandas_df)) # print the number of rows in the original dataset

pandas_df = pandas_df.sample(frac=0.5)
# Now print the number of rows in pandas_df to verify the
# downsampling
print(len(pandas_df))

# now split the data into independent variables (features)
# and a dependent variable (target)
X = pandas_df.drop('soda', axis=1)
X.head(n=3) # verify
y = pandas_df['soda']
y.head(n=3) # verify

# Convert "no" to 0 and "yes" to 1
y = y.map({"no": 0, "yes": 1})
y.head(n=3)



# now split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print(len(X_train)) # verify
print(len(y_train))
print(len(X_test))
print(len(y_test))



#
# Now it's time for one-hot-encoding.
#

# Identify categorical columns
categorical_cols = ['color']
numerical_cols = ['age', 'height', 'date']

# Create an encoder for categorical columns
# sparse_output=False allows the output to be transformed from a numpy array
# back to a pandas df
cat_encoder = OneHotEncoder(handle_unknown="ignore", 
                            sparse_output=False)

# Create a column transformer that applies different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', cat_encoder, categorical_cols)
    ],
    verbose_feature_names_out=False)

# Fit and transform the training data
preprocessor.set_output(transform="pandas")
X_train = preprocessor.fit_transform(X_train)
X_train.head(n=3)

# Transform the test data
X_test = preprocessor.transform(X_test)
X_test.head(n=3)

#
# Now it's time for standardizing the data if that
# is something we need to do. Some ML methods need this
# (like Support vector Machines and neural networks)
# others (like decision trees) do not
#
# Standardizing the data means subtracing the mean
# from each value and dividing by the standard deviation
#
# For this example, let's assume it is something
# we need to do. So we'll apply it to the
# "height" column

# First, create the scaler, which will fit and transform
# the training data...
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_train_scaled['height'] = scaler.fit_transform(
    X_train_scaled[['height']])
X_train_scaled.head(n=3)

# then apply that scaler to testing data (using the same mean and standard deviation
# that we calculated from the training data)
X_test_scaled = X_test.copy()
X_test_scaled['height'] = scaler.transform(
    X_test[['height']])
X_test_scaled.head(n=3)

# bam!

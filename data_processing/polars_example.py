import datetime

import polars as pl

path = "~/My Drive/stat_quests/course_intro_to_ml_2026/00_guides/code_and_stuff/"

polars_df = pl.read_csv(path + "example_data.txt", separator="\t")
# polars_df = pl.read_csv(path + "example_data.txt", 
#                         separator="\t",
#                         has_header=False)
# print out first 5 rows
polars_df.head()
# print out first 3 rows
polars_df.head(n=3)

# add column names
polars_df.columns = ["ID", 
                     "age", 
                     "height", 
                     "col", 
                     "name", 
                     "date", 
                     "soda"]
polars_df.head(n=3)

# print out column names
print(polars_df.columns)
# change "col" to "color"
polars_df = polars_df.rename({'ID' : 'uniqueID',
                              'col' : 'color'})
polars_df.head(n=3)

# Drop uniformitive columns
polars_df = polars_df.drop('uniqueID', 'name')
polars_df.head(n=3)



## convert the missing value "NA" to None, which is the value for null
## step 1: 
polars_df = polars_df.with_columns(
    # Start a conditional: test whether each value in "age" is the string "NA"
    pl.when(pl.col("age").is_in(["NA"]))
    # If the test is true, replace the value with a real null (missing value)
    .then(None)
    # If the test is false, keep the original "age" value (still a string)
    .otherwise(pl.col("age"))
    # Name the result "age" so it overwrites the existing column
    .alias("age")
)
polars_df.head(n=3)

## step 2: cast the "age" values from string to 64-bit integer (nulls stay null)
polars_df = polars_df.with_columns(
    pl.col("age").cast(pl.Int64)
)
polars_df.head(n=3)

## now let's convert the ??? in the height column to None...
polars_df = polars_df.with_columns(
    pl.when(pl.col("height").is_in(["???"]))
    .then(None)
    .otherwise(pl.col("height"))
    .alias("height")
)
polars_df.head(n=3)

## now let's convert the dtype for the height column to Float64
polars_df = polars_df.with_columns(
    pl.col("height").cast(pl.Float64)
)
polars_df.head(n=3)


## the values in the "color" column
## appear to have all kinds of capitalization
## so we print out the unique values...
print(polars_df["color"].unique())
## now make all the values in the "color" column lowercase
polars_df = polars_df.with_columns(
    pl.col("color")
    .str.to_lowercase()
    .alias("color") # overwrite existing column
)
## and verify the changes
print(polars_df["color"].unique())
polars_df.head(n=3)




## Now we need to fix the date values in the "date" column
## Step 1: remove any whitespace...
polars_df = polars_df.with_columns(
    pl.col("date").str.strip_chars()
)
## Step 2: parse each value on its own, since the formats are mixed (assumes month-first)
## Note: The order of the formats matters!!!
## When we do the conversions, "coalesce()" will try the formats 
## until it gets match and and then ignore everything that comes after.
## Thus, we try to match two-digit-year formats (%y) first before
## four-digit-year formats (%Y) so "1/3/19" isn't read as the (four-digit) year 19 AD
##
## Note: two-digit years 69–99 become 19xx and 00–68 become 20xx, so 5/4/73 is 1973.
## No future-date fix is needed. This is different from the pandas version.
## Also note: If you have two-digit years in the 2069+ range, they will be read wrongly
formats = ["%m/%d/%y", 
            "%m/%d/%Y", 
            "%Y/%m/%d", 
            "%m-%d-%Y", 
            "%b. %d '%y", 
            "%Y-%m-%d"]

polars_df = polars_df.with_columns(
    # For each row, take the first non-null result from the list of parsed columns
    ## The idea here is for coalesce to try each format in "formats" in order
    ## and the first format that succeeds, wins. Later formats will be ignored
    ## for that value.
    pl.coalesce(
        # Unpack the list of expressions into separate arguments for coalesce
        *[
            # Try to parse "date" as a Date with format f; failures become null
            pl.col("date").str.to_date(format=f, strict=False)
            # Build one parsing expression for each format in the list
            for f in formats
        ]
    # Name the result "date" so it overwrites the existing column
    ).alias("date")
)
# check the results
polars_df.head(n=3)

# now make sure that the reformatting didn't change 2 digit
# dates into furture dates. For example, '73 needs to be 1973
# (in the past) rather than 2073 (in the future)
today = datetime.date.today()

# find row indexes with future dates
future_idx = (
    polars_df
    .with_row_index("row_nr")
    .filter(pl.col("date") > today)
    .get_column("row_nr")
)

## print the first 3 examples of future dates
polars_df[future_idx.to_list()].head(n=3)

## correct the future dates
polars_df = polars_df.with_columns(
    pl.when(pl.col("date") > today)
    .then(pl.col("date").dt.offset_by("-100y"))
    .otherwise(pl.col("date"))
    .alias("date")
)

## print the first 3 examples of corrected future dates (now past)
polars_df[future_idx.to_list()].head(n=3)


## let's look at the different values in the "soda" column
print(polars_df["soda"].unique())
## and we see all kinds of different caplitalization...
## ...so let's make them all lowercase
polars_df = polars_df.with_columns(
    pl.col("soda").str.to_lowercase().alias("soda")
)
## verify
print(polars_df["soda"].unique())


polars_df.head(n=3)





## Now, just to show the method, we'll downsample the data...
print(len(polars_df))
downsampled_df = polars_df.sample(fraction=0.5)
## Now print the number of rows in pandas_df to verify the
## downsampling
print(len(downsampled_df))
print(downsampled_df)


## now all we have to do is convert the downsampled polars dataframe 
## into a pandas DataFrame.
pandas_df = downsampled_df.to_pandas()
pandas_df.head(n=3)

## BAM!!! We're done!!!
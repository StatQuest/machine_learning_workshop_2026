setwd(
  "~/My Drive/stat_quests/course_intro_to_ml_2026/00_guides/code_and_stuff/"
)

base_df <- read.delim("example_data.txt")
# base_df <- read.delim("example_data.txt", header = FALSE)
head(base_df, n = 3)
tail(base_df)
nrow(base_df)

base_df <- read.table(
  "example_data.txt",
  header = TRUE,
  sep = "\t",
  fill = TRUE,
  quote = "\"",
  comment.char = ""
)
head(base_df)
tail(base_df)
nrow(base_df)

# how to add column names...

# print out the first 6 rows
head(base_df)
# print out the first 3 rows
head(base_df, n = 3)

# print out column names
# names(base_df) <- c("ID",
#                     "age",
#                     "height",
#                     "col",
#                     "name",
#                     "date",
#                     "soda")
# names(base_df)
head(base_df, n = 3)

# change "ID" to "uniqueID"
names(base_df)[1] <- "uniqueID"
# change "col" to "color"
names(base_df)[4] <- "color"
head(base_df, n = 3)

# Drop uniformitive columns
base_df <- subset(base_df, select = -c(uniqueID, name))
head(base_df, n = 3)

# print out the data types for each column.
str(base_df)

# R correctly interpreted NA to be a missing value for
# age and its data type is what we expect, an integer.

# However, height has "???" for missing data and R
# converted the entire column to "chr", for character strings.
# So we need to fix this column by replacing "???" with NA
base_df$height <-
  ifelse(base_df$height == "???", NA, base_df$height)
head(base_df, n = 3)
str(base_df) # we see the data type is still "chr"

## now convert the datatype for the "height" column to numeric
base_df$height <- as.numeric(base_df$height)
str(base_df)

## Both "color" and "soda" should be factors and
## we need correct the random capitalization.
## However, first let's look at the different values
## in the "color" column
unique(base_df$color)
## And we see all kinds of different caplitalization,
## so let's make all of the strings lowercase
base_df$color <- tolower(base_df$color)
unique(base_df$color)
head(base_df, n = 3)
## Now let's see what the data types are...
str(base_df)
## ...and we see that "color" is still
## "chr", but we want it to be a factor, so
## let's convert it.
base_df$color <- as.factor(base_df$color)
## Now verify the change:
str(base_df)

# Load lubridate, which has functions for parsing dates
library(lubridate)

## NOTE: With lubridate, two-digit years 69–99 become 19xx and 00–68 become 20xx, so 5/4/73 is 1973.
## and '55 becomes 2055.

base_df$date <- as.Date(
  # as.Date() converts the date-times to plain dates (no time of day)

  # Try each order in turn for every value: month-day-year, "mdy", year-month-day, "ymd"
  # then abbreviated month name-day-year (e.g. "Jan. 1 '14"), "bdy"
  #
  # trimws() first removes any leading or trailing whitespace
  parse_date_time(trimws(base_df$date), orders = c("mdy", "ymd", "bdy"))
)

## Two-digit years can be parsed into the wrong century. Any date in the
## future is impossible here, so subtract 100 years (e.g. 2055 -> 1955).
## which() skips NA dates.
future_rows <- which(base_df$date > Sys.Date())
# head(future_rows)
base_df[head(future_rows, n = 3), ]
year(base_df$date[future_rows]) <- year(base_df$date[future_rows]) - 100
base_df[head(future_rows, n = 3), ]


# correct inconsistent capitalization for soda
print(unique(base_df$soda))
## and we see all kinds of different caplitalization...
## ...so let's make them all lowercase
base_df$soda <- tolower(base_df$soda)

## let's check the updated values
print(unique(base_df$soda))
## and just look at the new values
head(base_df, n = 3)
str(base_df)
base_df$soda <- as.factor(base_df$soda)
str(base_df)

## Now, just to show the method, we'll downsample the data...
nrow(base_df)
downsampled.base_df <- base_df[
  sample(
    nrow(base_df),
    size = floor(nrow(base_df) * 0.5)
  ),
]
## Now print the number of rows in pandas_base_df to verify the
## downsampling
nrow(downsampled.base_df)

## Most programs in R do not require you to isoluate the
## dependent variable from everything else. However, if you
## Want to use XGBoost, you'll need to do it. Here's how.
X <- subset(base_df, select = -c(soda))
head(X, n = 3)
y <- base_df$soda
y <- ifelse(base_df$soda == "yes", 1, 0)
head(y, n = 3)

## now split the data into train and test sets
## In this example, we're just taking the first 80% of the rows
## for training with head()
train.base_df <- head(
  downsampled.base_df,
  n = 0.8 * nrow(downsampled.base_df)
)
nrow(train.base_df)
## and the last 20% of the rows for testing with tail()
test.base_df <- tail(
  downsampled.base_df,
  n = 0.2 * nrow(downsampled.base_df)
)
nrow(test.base_df)

##
## NOTE: In R, we don't need to one-hot encode the data
##

##
## Now it's time for standardizing the data if that
## is something we need to do. Some ML methods need this
## (like Support vector Machines and neural networks)
## others (like decision trees) do not
##
## Standardizing the data means subtracing the mean
## from each value and dividing by the standard deviation
##
## For this example, let's assume it is something
## we need to do. So we'll apply it to the
## "height" column
head(train.base_df, n = 3)
train.height.mean <- mean(train.base_df$height, na.rm = TRUE)
train.height.mean
train.height.sd <- sd(train.base_df$height, na.rm = TRUE)
train.height.sd

## now standardize...
train.base_df$height <-
  (train.base_df$height - train.height.mean) /
  train.height.sd
head(train.base_df, n = 3)
summary(train.base_df$height)

test.base_df$height <-
  (test.base_df$height - train.height.mean) /
  train.height.sd
head(test.base_df, n = 3)
summary(test.base_df$height)

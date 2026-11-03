library(tidyverse) # readr, dplyr, tibble, forcats, stringr, lubridate, ...

# Set the working directory
setwd(
  "~/My Drive/stat_quests/course_intro_to_ml_2026/00_guides/code_and_stuff/"
)

tidy_df <- read_tsv("example_data.txt")
# print out the first 6 rows
tidy_df |> head()
# print out the first 3 rows
tidy_df |> slice_head(n = 3)

# print out column names
tidy_df |> colnames()
# change "col" to "color"
tidy_df <- tidy_df |> rename(color = col)
tidy_df |> colnames()

# Drop uninformative columns
tidy_df <- tidy_df |> select(-c(uniqueID, name))
tidy_df |> slice_head(n = 3)

# readr correctly interpreted NA to be a missing value for
# age and its data type is what we expect, an integer.

# However, height has "???" for missing data and readr
# kept the entire column as "chr", for character strings.
# So we need to fix this column by replacing "???" with NA
tidy_df <- tidy_df |> mutate(height = na_if(height, "???"))
tidy_df |> slice_head(n = 3)

## now convert the datatype for the "height" column to numeric
tidy_df <- tidy_df |> mutate(height = as.numeric(height))
tidy_df |> slice_head(n = 3)

## Both "color" and "soda" should be factors and
## we need correct the random capitalization.
## However, first let's look at the different values
## in the "color" column
tidy_df |> distinct(color)
## And we see all kinds of different capitalization,
## so let's make all of the strings lowercase
tidy_df <- tidy_df |> mutate(color = str_to_lower(color))
tidy_df |> distinct(color)
tidy_df |> slice_head(n = 3)
## ...and we see that "color" is still
## "chr", but we want it to be a factor, so
## let's convert it.
tidy_df <- tidy_df |> mutate(color = as_factor(color))
## Now verify the change:
tidy_df |> slice_head(n = 3)

# now do the same thing for soda.
tidy_df |> distinct(soda)
## ...so let's make them all lowercase and convert to a factor
tidy_df <- tidy_df |>
  mutate(soda = soda |> str_to_lower() |> as_factor())

## let's check the updated values
tidy_df |> distinct(soda)
tidy_df |> slice_head(n = 3)

## NOTE: With lubridate, two-digit years 69-99 become 19xx and 00-68 become 20xx, so 5/4/73 is 1973.
## No future-date fix is needed. This is the same as the polars version and different from the pandas version.
## ALSO NOTE: If you have two-digit years in the 2069+ range, they will be read wrongly

tidy_df <- tidy_df |>
  mutate(
    # str_trim() first removes leading/trailing whitespace.
    # Then try each order in turn for every value: month-day-year "mdy",
    # year-month-day "ymd", then abbreviated month name-day-year
    # (e.g. "Jan. 1 '14") "bdy".
    # Finally, as_date() converts the date-times to plain dates.
    date = date |>
      str_trim() |>
      parse_date_time(orders = c("mdy", "ymd", "bdy")) |>
      as_date()
  )

# Check the new values and the new column type
tidy_df |> slice_head(n = 10)

## Now, just to show the method, we'll downsample the data...
downsampled_tidy_df <- tidy_df |> slice_sample(prop = 0.5)
## Now print the number of rows to verify the downsampling
downsampled_tidy_df |> nrow()

## now split the data into train and test sets
train_tidy_df <- downsampled_tidy_df |> slice_head(prop = 0.8)
train_tidy_df |> nrow()
test_tidy_df <- downsampled_tidy_df |> slice_tail(prop = 0.2)
test_tidy_df |> nrow()

##
## NOTE: In R, we don't need to one-hot encode the data
##

##
## Now it's time for standardizing the data if that
## is something we need to do. Some ML methods need this
## (like Support Vector Machines and neural networks)
## others (like decision trees) do not
##
## Standardizing the data means subtracting the mean
## from each value and dividing by the standard deviation
##
## For this example, let's assume it is something
## we need to do. So we'll apply it to the
## "height" column
train_tidy_df |> slice_head(n = 3)
train_height_stats <- train_tidy_df |>
  summarize(
    mean = mean(height, na.rm = TRUE),
    sd = sd(height, na.rm = TRUE)
  )
train_height_stats

## now standardize, using the training mean and sd for both sets...
train_tidy_df <- train_tidy_df |>
  mutate(height = (height - train_height_stats$mean) / train_height_stats$sd)
train_tidy_df |> slice_head(n = 3)
train_tidy_df$height |> summary()

test_tidy_df <- test_tidy_df |>
  mutate(height = (height - train_height_stats$mean) / train_height_stats$sd)
test_tidy_df |> slice_head(n = 3)
test_tidy_df$height |> summary()

##
## Summary: all of the cleaning steps above in a single pipeline
##
tidy_df_clean <- read_tsv("example_data.txt") |>
  rename(color = col) |>
  select(-c(uniqueID, name)) |>
  mutate(
    height = height |> na_if("???") |> as.numeric(),
    color = color |> str_to_lower() |> as_factor(),
    soda = soda |> str_to_lower() |> as_factor(),
    date = date |>
      str_trim() |>
      parse_date_time(orders = c("mdy", "ymd", "bdy")) |>
      as_date()
  )
tidy_df_clean |> slice_head(n = 5)

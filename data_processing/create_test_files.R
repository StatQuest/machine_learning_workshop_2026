setwd(
  "~/My Drive/stat_quests/course_intro_to_ml_2026/00_guides/code_and_stuff/"
)

set.seed(42)
num.rows <- 100

colors <- c("red", "blue", "green", "Green")
heights <- c(1.2, 1.6, "???", 1.5, 1.4, 1.3, 1.5)
names <- c(
  "Joe",
  "Sue",
  "Ron",
  "Ann",
  "Pete",
  "Beth",
  "Dave",
  "Sim",
  "Fred",
  "Tex",
  "Ric",
  "Tay",
  "Sid",
  "Sal",
  "Kim",
  "Suz"
)
responses <- c("Yes", "no", "yes", "Yes", "No", "No", "Yes")
dates <- c(
  "12/01/2022",
  "Jan. 1 '14",
  "2023/04/13",
  "12-1-2008",
  "1/3/19",
  "2/17/03",
  "5/4/73"
)

more.years <- sample(
  seq(from = 10, to = 90, by = 1),
  size = (num.rows - length(dates)),
  replace = TRUE
)
more.months <- sample(
  1:12,
  size = (num.rows - length(dates)),
  replace = TRUE
)
more.days <- sample(
  1:28,
  size = (num.rows - length(dates)),
  replace = TRUE
)

more.dates <- paste(more.months, more.days, more.years, sep = "/")
head(more.dates)

dates <- c(dates, as.character(more.dates))
length(dates)

df <- data.frame(
  ID = sample(seq(from = 1000, to = 10000, by = 1), size = num.rows),
  age = c(NA, round(rnorm(num.rows - 1, mean = 30, sd = 10))),
  height = c(
    1.6,
    1.5,
    "???",
    sample(heights, size = (num.rows - 3), replace = TRUE)
  ),
  col = sample(colors, size = num.rows, replace = TRUE),
  name = sample(names, size = num.rows, replace = TRUE),
  date = dates,
  soda = sample(responses, size = num.rows, replace = TRUE)
)
head(df)
str(df)


write.table(
  df,
  file = "example_data.txt",
  sep = "\t",
  row.names = FALSE,
  quote = FALSE,
  col.names = TRUE
)

# csv.df <- data.frame(
#   a = rnorm(n = 7, mean = 0, sd = 1),
#   b = c('comma', 'b', 'c', 'd', 'e', 'f', 'g'),
#   c = rnorm(n = 7, mean = 100, sd = 1)
# )
# csv.df

# write.csv(csv.df, file = "data.csv", row.names = FALSE, quote = FALSE)

# write.table(
#   csv.df,
#   file = "data_no_header.csv",
#   sep = ",",
#   row.names = FALSE,
#   quote = FALSE,
#   col.names = FALSE
# )

# #############
# # Now create a new data frame for tab-delimited files
# # NOTE: We just need something that is a little different
# # so I can tell csv and tab file apart when I open them.
# #############
# tab.df <- data.frame(
#   a = rnorm(n = 7, mean = 0, sd = 1),
#   b = c('tab', 'b', 'c', 'd', 'e', 'f', 'g'),
#   c = rnorm(n = 7, mean = 100, sd = 1)
# )
# tab.df

# write.table(
#   tab.df,
#   file = "data.txt",
#   row.names = FALSE,
#   sep = "\t",
#   quote = FALSE
# )

# write.table(
#   tab.df,
#   file = "data_no_header.txt",
#   row.names = FALSE,
#   sep = "\t",
#   quote = FALSE,
#   col.names = FALSE
# )

# head(tab.df)
# head(tab.df, n = 3)

# bad.data <- data.frame(
#   Age = c(12, NA, 13, 14),
#   Height = c(1.2, 1.6, "???", 1.5),
#   Date = c("12/01/2022", "Jan. 1 '14", "2023/04/13", "12-1-2008"),
#   Name = c("Joe", "Sue", "Ron", "Ann"),
#   Soda = c("Yes", "no", "yes", "Yes")
# )
# bad.data

# write.table(
#   bad.data,
#   file = "bad_data.txt",
#   row.names = FALSE,
#   sep = "\t",
#   quote = FALSE
# )

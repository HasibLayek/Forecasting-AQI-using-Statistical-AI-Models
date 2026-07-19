# ==============================================================================
# Exploratory Data Analysis & Imputation
# Source: Appendix A (Section A) — R Programming Scripts
# Project: Forecasting the Air Quality Index using Statistical, Machine
#          Learning and Deep Learning Models — A Comparative Study
# Department of Statistics & Operations Research, Aligarh Muslim University
# ==============================================================================

# ===============================
# 1. Data Understanding
# ===============================

library(tidyverse)
library(lubridate)

data <- read.csv("data/raw/DL017.csv")

str(data)
head(data)

data$From.Date <- dmy_hm(data$From.Date)
data$To.Date   <- dmy_hm(data$To.Date)

range(data$From.Date, na.rm = TRUE)


# ===============================
# 2. Data Quality Checks
# ===============================

colMeans(is.na(data)) * 100
sum(duplicated(data))
summary(data)

# Strip duplicated unit-suffixes from column names (CPCB export artifact)
names(data) <- gsub("\\.\\..*", "", names(data))
names(data)

pollutants <- data %>%
  select(PM2.5, PM10, NO, NO2, NOx, Ozone)

boxplot(pollutants, las = 2, col = "lightblue")

par(mfrow = c(2, 3))
for (col in names(pollutants)) {
  boxplot(pollutants[[col]], main = col, col = "lightgreen")
}
par(mfrow = c(1, 1))


# ===============================
# 3. Univariate Analysis
# ===============================

hist(data$PM2.5, main = "PM2.5 Distribution", xlab = "PM2.5")

plot(data$From.Date, data$PM2.5, type = "l",
     col = "blue", main = "PM2.5 over Time")

library(zoo)
data$PM2.5_rollmean <- rollmean(data$PM2.5, k = 7, fill = NA)

plot(data$From.Date, data$PM2.5_rollmean,
     type = "l", col = "red",
     main = "7-day Rolling Mean of PM2.5")


# ===============================
# 4. Bivariate & Multivariate Analysis
# ===============================

cor_matrix <- cor(pollutants, use = "pairwise.complete.obs")
print(cor_matrix)

library(corrplot)
corrplot(cor_matrix, method = "color",
         tl.col = "black", addCoef.col = "white")

plot(data$PM2.5, data$PM10,
     main = "PM2.5 vs PM10",
     xlab = "PM2.5", ylab = "PM10")


# ===============================
# 5. Imputation (month x hour group-mean imputation)
# ===============================
# Missing values are filled using the mean for the same calendar
# month + hour-of-day combination, falling back to the global mean
# where a given (month, hour) group has no observed values at all.

data <- data %>%
  arrange(From.Date) %>%
  mutate(
    month = month(From.Date),
    hour  = hour(From.Date)
  )

exclude <- c("From.Date", "To.Date", "month", "hour")
numeric_cols <- names(data)[sapply(data, is.numeric) &
                               !(names(data) %in% exclude)]

global_means <- sapply(data[numeric_cols],
                        function(x) mean(x, na.rm = TRUE))

for (col in numeric_cols) {

  was_na_col <- paste0(col, "_was_na")
  imp_col    <- paste0(col, "_imputed")

  data[[was_na_col]] <- is.na(data[[col]])

  grp_means <- data %>%
    group_by(month, hour) %>%
    summarise(gmean = mean(.data[[col]], na.rm = TRUE),
              .groups = "drop")

  data <- data %>%
    left_join(grp_means, by = c("month", "hour")) %>%
    mutate(
      temp = ifelse(is.na(.data[[col]]), gmean, .data[[col]]),
      temp = ifelse(is.na(temp), global_means[col], temp),
      !!imp_col := temp
    ) %>%
    select(-gmean, -temp)
}

write.csv(data, "data/processed/DL017_imputed_monthXhour.csv", row.names = FALSE)


# ===============================
# 6. Data Visualisation
# ===============================

data$year  <- year(data$From.Date)
data$month <- month(data$From.Date)

monthly_avg <- data %>%
  group_by(year, month) %>%
  summarise(mean_pm25 = mean(PM2.5, na.rm = TRUE))

ggplot(monthly_avg,
       aes(x = month, y = mean_pm25,
           group = year, color = factor(year))) +
  geom_line() + geom_point() +
  labs(title = "Monthly Average PM2.5")

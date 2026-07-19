# ==============================================================================
# Prophet Modeling (Daily Data)
# Source: Appendix A (Section C) — R Programming Scripts
# Project: Forecasting the Air Quality Index using Statistical, Machine
#          Learning and Deep Learning Models — A Comparative Study
# ==============================================================================

library(readr)
library(dplyr)
library(lubridate)
library(zoo)
library(prophet)
library(ggplot2)

raw <- read_csv("data/processed/DL017_Lodhi_road.csv", show_col_types = FALSE)

raw <- raw %>%
  mutate(From.Date = parse_date_time(
    From.Date,
    orders = c("d-m-Y H:M", "d-m-Y H:M:S", "d-m-Y")
  )) %>%
  filter(!is.na(From.Date)) %>%
  mutate(date = as_date(From.Date))

daily <- raw %>%
  group_by(date) %>%
  summarise(
    PM25 = mean(PM2.5_imputed, na.rm = TRUE),
    PM10 = mean(PM10_imputed, na.rm = TRUE),
    .groups = "drop"
  )

full_dates <- tibble(date = seq(min(daily$date), max(daily$date), by = "day"))

daily <- full_dates %>%
  left_join(daily, by = "date") %>%
  mutate(across(c(PM25, PM10), ~ na.approx(.x, rule = 2)))

last3 <- daily %>%
  filter(date >= max(date) - years(3))


# -------------------------------
# Prophet Forecast Function
# -------------------------------

run_fc <- function(df, varname) {

  df_prophet <- df %>%
    select(ds = date, y = {{ varname }})

  m <- prophet(df_prophet)
  future <- make_future_dataframe(m, periods = 365)
  fc <- predict(m, future)

  ggplot(fc, aes(ds, yhat)) +
    geom_line(color = "darkgreen") +
    geom_ribbon(aes(ymin = yhat_lower, ymax = yhat_upper), alpha = 0.2) +
    labs(
      title = paste(varname, "1-year Prophet Forecast"),
      x = "Date", y = "\u00b5g/m\u00b3"
    )
}

par(mfrow = c(2, 2))
run_fc(last3, "PM25")
run_fc(last3, "PM10")


# ===============================
# Prophet Model Accuracy
# ===============================
# 80/20 chronological split (no shuffling), evaluated on RMSE, MAE, MAPE.

library(Metrics)

prophet_accuracy <- function(df, colname) {

  data <- df %>%
    select(ds = date, y = all_of(colname)) %>%
    mutate(ds = as.Date(ds)) %>%
    filter(!is.na(y))

  n <- nrow(data)
  train_n <- floor(0.8 * n)

  train <- data[1:train_n, ]
  test  <- data[(train_n + 1):n, ]

  m <- prophet(train)
  future <- make_future_dataframe(m, periods = nrow(test))

  fc <- predict(m, future) %>%
    mutate(ds = as.Date(ds))

  joined <- left_join(test, fc, by = "ds")

  rmse_val <- rmse(joined$y, joined$yhat)
  mae_val  <- mae(joined$y, joined$yhat)
  mape_val <- mean(abs((joined$y - joined$yhat) / pmax(joined$y, 1e-6))) * 100

  cat("\nAccuracy for ", colname, ":\n")
  cat("RMSE: ", rmse_val, "\n")
  cat("MAE: ", mae_val, "\n")
  cat("MAPE: ", mape_val, "%\n")

  ggplot() +
    geom_line(data = train, aes(ds, y), color = "black") +
    geom_line(data = test, aes(ds, y), color = "blue") +
    geom_line(data = joined, aes(ds, yhat), color = "red") +
    labs(
      title = paste(colname, "Prophet Train/Test Performance"),
      x = "Date", y = "\u00b5g/m\u00b3"
    )
}

prophet_accuracy(last3, "PM25")
prophet_accuracy(last3, "PM10")

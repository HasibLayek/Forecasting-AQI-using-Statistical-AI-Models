# ==============================================================================
# SARIMA Modeling (Daily Data)
# Source: Appendix A (Section B) — R Programming Scripts
# Project: Forecasting the Air Quality Index using Statistical, Machine
#          Learning and Deep Learning Models — A Comparative Study
# ==============================================================================

# ===============================
# Convert Hourly to Daily Data
# ===============================

library(dplyr)
library(lubridate)
library(zoo)
library(ggplot2)
library(tseries)
library(forecast)

imp_data <- read.csv("data/processed/DL017_Lodhi_road.csv",
                      stringsAsFactors = FALSE)

imp_data <- imp_data %>%
  mutate(
    datetime = parse_date_time(
      From.Date,
      orders = c("dmy HMS", "dmy HM",
                 "dmy H", "dmy"),
      tz = "UTC"
    ),
    date = as.Date(datetime)
  )

daily_data <- imp_data %>%
  group_by(date) %>%
  summarise(
    mean_pm25 = mean(PM2.5_imputed, na.rm = TRUE),
    mean_pm10 = mean(PM10_imputed, na.rm = TRUE)
  ) %>%
  arrange(date) %>%
  ungroup()

head(daily_data)


# ===============================
# Time Series Objects
# ===============================

ts_pm25 <- ts(daily_data$mean_pm25,
              frequency = 365,
              start = c(2017, 1))

ts_pm10 <- ts(daily_data$mean_pm10,
              frequency = 365,
              start = c(2017, 1))


# ===============================
# Stationarity Check: PM2.5
# ===============================

adf.test(ts_pm25)
kpss.test(ts_pm25)

acf(ts_pm25, main = "ACF of PM2.5 (Daily)")
pacf(ts_pm25, main = "PACF of PM2.5 (Daily)")


# ===============================
# Stationarity Check: PM10
# ===============================

adf.test(ts_pm10)
kpss.test(ts_pm10)

acf(ts_pm10, main = "ACF of PM10 (Daily)")
pacf(ts_pm10, main = "PACF of PM10 (Daily)")


# ===============================
# Raw Time Series Plot (PM2.5)
# ===============================

ggplot(daily_data, aes(x = date, y = mean_pm25)) +
  geom_line(color = "steelblue", linewidth = 0.8) +
  labs(
    title = "Daily Mean PM2.5 Concentration (2017-2024)",
    subtitle = "Location: Lodhi Road (DL017)",
    x = "Date",
    y = expression(paste("PM"[2.5], " (", mu, "g/m"^3, ")"))
  ) +
  theme_minimal()


# ===============================
# SARIMA Forecasting
# ===============================
# Orders are auto-selected by auto.arima() (seasonal period = 365,
# forced seasonal differencing D = 1) rather than hand-specified.

fit_sarima_pm25 <- auto.arima(ts_pm25, seasonal = TRUE, D = 1)
forecast_pm25 <- forecast(fit_sarima_pm25, h = 365)

fit_sarima_pm10 <- auto.arima(ts_pm10, seasonal = TRUE, D = 1)
forecast_pm10 <- forecast(fit_sarima_pm10, h = 365)


# ===============================
# Actual vs Forecast Plots
# ===============================

combined_pm25 <- bind_rows(
  daily_data %>%
    select(date, mean_pm25) %>%
    rename(Value = mean_pm25) %>%
    mutate(Type = "Actual"),
  data.frame(
    date = seq(max(daily_data$date) + 1, by = "day", length.out = 365),
    Value = as.numeric(forecast_pm25$mean),
    Type = "Forecast"
  )
)

ggplot(combined_pm25, aes(x = date, y = Value, color = Type)) +
  geom_line(linewidth = 0.8) +
  labs(
    title = "Actual vs Forecasted PM2.5 (SARIMA)",
    x = "Date",
    y = "PM2.5 (\u00b5g/m\u00b3)"
  ) +
  theme_minimal()


combined_pm10 <- bind_rows(
  daily_data %>%
    select(date, mean_pm10) %>%
    rename(Value = mean_pm10) %>%
    mutate(Type = "Actual"),
  data.frame(
    date = seq(max(daily_data$date) + 1, by = "day", length.out = 365),
    Value = as.numeric(forecast_pm10$mean),
    Type = "Forecast"
  )
)

ggplot(combined_pm10, aes(x = date, y = Value, color = Type)) +
  geom_line(linewidth = 0.8) +
  labs(
    title = "Actual vs Forecasted PM10 (SARIMA)",
    x = "Date",
    y = "PM10 (\u00b5g/m\u00b3)"
  ) +
  theme_minimal()


# ===============================
# Model Accuracy & Information Criteria
# ===============================

accuracy(fit_sarima_pm25)
accuracy(fit_sarima_pm10)

AIC(fit_sarima_pm25)
BIC(fit_sarima_pm25)

AIC(fit_sarima_pm10)
BIC(fit_sarima_pm10)


# ===============================
# Save Models
# ===============================

# saveRDS(fit_sarima_pm25, "results/sarima_pm25_model.rds")
# saveRDS(fit_sarima_pm10, "results/sarima_pm10_model.rds")

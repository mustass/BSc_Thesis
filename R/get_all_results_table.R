# results table: 
rm(list = ls())
library(dplyr)
library(xtable)
#One-step Stats:

arima_volatility_preest <- read.table("/home/s/Dropbox/KU/BSc Stas/R/ARIMA_final_results/preEst/volatility_preest_table.txt",header = T)
arima_returns_preest <- read.table("/home/s/Dropbox/KU/BSc Stas/R/ARIMA_final_results/preEst/returns_preest_table.txt",header = T)


# One-step ML:


lstm_volatility_preest <- read.table("/home/s/Dropbox/KU/BSc Stas/R/ML_final_results/preEst/simple_backtest_output_table_vol.txt",header = T)
lstm_returns_preest <- read.table("/home/s/Dropbox/KU/BSc Stas/R/ML_final_results/preEst/simple_backtest_output_table.txt",header = T)

table_one_step <- arima_returns_preest %>% mutate(type = "ret") %>%  rbind(arima_volatility_preest %>% mutate(type = "vol") ) %>% 
  mutate(arma_rmse=RMSE) %>% as.data.frame() %>%  select(-RMSE) %>% left_join(
    lstm_returns_preest %>% mutate(type = "ret") %>% rbind(lstm_volatility_preest %>% mutate(type = "vol") ) %>% mutate(lstm_rmse=RMSE) %>% as.data.frame() %>% 
      select(-RMSE) ,by = c("Series", "type")
  ) %>% mutate(diff = arma_rmse-lstm_rmse) %>% mutate(pct = diff*100/arma_rmse)

xtable(table_one_step %>% select(-type),digits = -6)


# Rolling

# ML:

lstm_returns_rolling_1yr <- read.table("/home/s/Dropbox/KU/BSc Stas/R/ML_final_results/rolling_window_1/rolling_1_returns_output_table.txt",header = T)%>% mutate(type= "ret")
lstm_volatility_rolling_1yr <- read.table("/home/s/Dropbox/KU/BSc Stas/R/ML_final_results/rolling_window_1/rolling_1_volatility_output_table.txt",header = T)%>% mutate(type= "vol")

lstm_returns_rolling_3yr <- read.table("/home/s/Dropbox/KU/BSc Stas/R/ML_final_results/rolling_window_3/rolling_3_returns_output_table.txt",header = T)%>% mutate(type= "ret")
lstm_volatility_rolling_3yr <- read.table("/home/s/Dropbox/KU/BSc Stas/R/ML_final_results/rolling_window_3/rolling_3_volatility_output_table.txt",header = T)%>% mutate(type= "vol")

# ARIMA


arima_volatility_rolling_1<- read.table("/home/s/Dropbox/KU/BSc Stas/R/ARIMA_final_results/rolling_1/volatility_rolling_1.txt",header = T) %>%  mutate(type= "vol")
arima_returns_rolling_1 <- read.table("/home/s/Dropbox/KU/BSc Stas/R/ARIMA_final_results/rolling_1/returns_rolling_1yr_table.txt",header = T)%>% mutate(type= "ret")

arima_volatility_rolling_3<- read.table("/home/s/Dropbox/KU/BSc Stas/R/ARIMA_final_results/rolling_3/volatility_rolling_3yr.txt",header = T) %>%  mutate(type= "vol")
arima_returns_rolling_3 <- read.table("/home/s/Dropbox/KU/BSc Stas/R/ARIMA_final_results/rolling_3/returns_rolling_3yr_table.txt",header = T)%>% mutate(type= "ret")


table_rolling_1yr <- arima_returns_rolling_1 %>% mutate(type = "ret") %>% rbind(arima_volatility_rolling_1 %>% mutate(type = "vol")) %>% 
  mutate(arma_rmse=RMSE) %>% as.data.frame() %>%  select(-RMSE) %>% left_join(
    lstm_returns_rolling_1yr %>% mutate(type = "ret") %>% rbind(lstm_volatility_rolling_1yr %>% mutate(type = "vol") ) %>% mutate(lstm_rmse=RMSE) %>% as.data.frame() %>% 
    select(-RMSE) ,by = c("Series", "type")
) %>% mutate(diff = arma_rmse-lstm_rmse) %>% mutate(pct = diff*100/arma_rmse)


table_rolling_3yr <- arima_returns_rolling_3 %>% mutate(type = "ret") %>% rbind(arima_volatility_rolling_3 %>% mutate(type = "vol")) %>% 
  mutate(arma_rmse=RMSE) %>% as.data.frame() %>%  select(-RMSE) %>% left_join(
    lstm_returns_rolling_3yr %>% mutate(type = "ret") %>% rbind(lstm_volatility_rolling_3yr %>% mutate(type = "vol") ) %>% mutate(lstm_rmse=RMSE) %>% as.data.frame() %>% 
      select(-RMSE) ,by = c("Series", "type")
  ) %>% mutate(diff = arma_rmse-lstm_rmse) %>% mutate(pct = diff*100/arma_rmse)

# Now to the window tables: 

last_table <- table_one_step %>% select(-c(pct,diff)) %>% mutate(arma_rmse_one_step = arma_rmse,
                                                                 lstm_rmse_one_step = lstm_rmse) %>% select(-c(arma_rmse,lstm_rmse)) %>% 
  left_join(table_rolling_1yr %>% select(-c(pct,diff)) %>% mutate(arma_rmse_rolling_1 = arma_rmse,
                                                               lstm_rmse_rolling_1 = lstm_rmse) %>% select(-c(arma_rmse,lstm_rmse)),by = c("Series","type")  
            ) %>% 
  left_join(table_rolling_3yr %>% select(-c(pct,diff)) %>% mutate(arma_rmse_rolling_3 = arma_rmse,
                                                                  lstm_rmse_rolling_3 = lstm_rmse) %>% select(-c(arma_rmse,lstm_rmse)),by = c("Series","type")  
  ) %>% rowwise() %>% 
  mutate(best =min(arma_rmse_one_step, lstm_rmse_one_step ,arma_rmse_rolling_1 ,lstm_rmse_rolling_1 ,arma_rmse_rolling_3 ,lstm_rmse_rolling_3)) %>% 
  mutate(worst =max(arma_rmse_one_step, lstm_rmse_one_step ,arma_rmse_rolling_1 ,lstm_rmse_rolling_1 ,arma_rmse_rolling_3 ,lstm_rmse_rolling_3)) %>% 
  as.data.frame()

xtable(last_table, digits = -5)
xtable(table_rolling_3yr %>% select(-type),digits = -6)

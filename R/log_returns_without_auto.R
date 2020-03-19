### Backtesting m8
rm(list = ls())
library(forecast)
library(tseries)
require(knitr)
library(fGarch)
library(xts)
library(tidyverse)
library(lubridate)
library(FinAna)
library(lubridate)
library(quantmod)
library(fpp2)
library(DT)
library(ggfortify)
library(psych)
library(xtable)
require(gridExtra)
setwd("/home/s/Dropbox/KU/BSc Stas/R")
save_path <- "/home/s/Dropbox/KU/BSc Stas/R/10March_returns/one_step_2015"
### Data prep:
DJI_csv <-  as.xts(read.csv.zoo("data/data_returns/DJI.csv"))
GSPC_csv <- as.xts(read.csv.zoo("data/data_returns/GSPC.csv"))
IXIC_csv <- as.xts(read.csv.zoo("data/data_returns/IXIC.csv"))
N225_csv <- as.xts(read.csv.zoo("data/data_returns/N225.csv"))
summary(N225_csv)
storage.mode(N225_csv) <- "numeric"


GSPC_log <- log(GSPC_csv)
DJI_log <- log(DJI_csv)
IXIC_log <- log(IXIC_csv)
N225_log <- log(N225_csv)


GSPC_returns<- diff(GSPC_log$Adj.Close) %>% na.approx()
GSPC_returns <- GSPC_returns[index(GSPC_returns)>="1985-01-01"]
DJI_returns <-  diff(DJI_log$Adj.Close) %>% na.approx()
DJI_returns <- DJI_returns[index(DJI_returns)>="1985-01-01"]
IXIC_returns <-  diff(IXIC_log$Adj.Close) %>% na.approx()
IXIC_returns <- IXIC_returns[index(IXIC_returns)>="1985-01-01"]
N225_returns <-  diff(N225_log$Adj.Close) %>% na.approx()
N225_returns <- N225_returns[index(N225_returns)>="1985-01-01"]


model_gspc <- auto.arima(GSPC_returns[index(GSPC_returns)<"2014-01-01"], stepwise = FALSE, method = "ML")
model_dji <- auto.arima(DJI_returns[index(DJI_returns)<"2014-01-01"], stepwise = FALSE, method = "ML")
model_ixic <- auto.arima(IXIC_returns[index(IXIC_returns)<"2014-01-01"], stepwise = FALSE, method = "ML")
model_n225 <- auto.arima(N225_returns[index(N225_returns)<"2014-01-01"], stepwise = FALSE, method = "ML")

summary(model_gspc)
summary(model_dji)
summary(model_ixic)
summary(model_n225)

# Backtesting functions:
one_step_no_rest <- function(dataset, start){
  train <- dataset[index(dataset)< start]
  test <- dataset[index(dataset)>= start] %>% head(-1)
  print(length(test))
  model <- auto.arima(train, method = "ML",stepwise = FALSE)
  summary(model)
  test_set_model_arma <- Arima(test, model=model)

  predictions_dt <-timeSeries(test_set_model_arma$fitted,index(test)) %>% as.xts()
  length(predictions_dt)
  plot <- ggplot() + 
    geom_line(data = test, aes(x = index(test), y = test, color = "red"),size =0.5) +
    geom_line(data = predictions_dt, aes(x = index(test), y = predictions_dt, color = "darkblue"),size = 0.5) +
    xlab('Time') +
    ylab('Log-returns')+scale_color_discrete(name = "Series", labels = c("Predicted","Actual"))
  
  
  manual_rmse <- Metrics::rmse(as.numeric(test$Adj.Close),as.numeric(predictions_dt$TS.1))
  
  automatic_rmse_train <- accuracy(model)[2]
  print(automatic_rmse_train)
  automatic_rmse <- accuracy(test_set_model_arma)[2]
  output <- list(manual_rmse,automatic_rmse, plot)
  return(output)
}

one_step_w_rest <- function(dataset, arma, window_length,auto, eval_start){

  if (window_length == 3){
    dataset <- dataset[index(dataset) >="2012-01-01"]
  }
  if (window_length == 1){
    dataset <- dataset[index(dataset)>="2014-01-01",]
  }

  start = as.Date(index(dataset)[1])
  end = index(dataset)[253*window_length]
  end_of_loop <- length(dataset)-253*window_length+1
  test <- xts::last(dataset, end_of_loop-1)
  
  preds <- c()
  labels <- dataset[index(dataset) >=index(dataset)[253*window_length+1]]
  i = 1
  while(i < end_of_loop){
    #print(i)
    train <- window(dataset,  start = start, end = end)
    
    if(auto){
      model <- auto.arima(train, stepwise = FALSE, method = "ML")
    }
    else{
      model <- Arima(train, order = arma, method = "ML")
      }
    preds[i] <- forecast::forecast(model)$mean[1]
    
    
    # NEXT:
    i = i+1
    start = index(dataset)[i]
    end = index(dataset)[253*window_length+i-1] # Check this mate!!
  }
  
  
  #predictions_dt <-data.frame(predict = preds) 
  plot_dt <- data.frame(Datetime = index(labels), preds = preds, true_vals = labels)%>% filter(Datetime >=eval_start) %>% head(-1)
  print(dim(plot_dt))
  plot <- ggplot(plot_dt, aes(x = Datetime)) + 
    geom_line(aes( y = Adj.Close, color = "red"),size = 0.5) +
    geom_line(aes( y = preds, color = "darkblue"),size = 0.5) +
    xlab('Time') +
    ylab('Log-returns')+scale_color_discrete(name = "Series", labels = c("Predicted","Actual"))
  #print(plot)
  loss <- Metrics::rmse(plot_dt$Adj.Close,plot_dt$preds)
  output <- list(loss,plot)
  return(output)
}

setwd(save_path)

# One-step without reestimation: 
DJI_simple_backtest <- one_step_no_rest(DJI_returns, "2015-03-09" )
GSPC_simple_backtest<- one_step_no_rest(GSPC_returns, "2015-03-05")
IXIC_simple_backtest<- one_step_no_rest(IXIC_returns, "2015-03-10")
N225_simple_backtest<- one_step_no_rest(N225_returns, "2015-02-16")


plot1 <- DJI_simple_backtest[[3]]  + ggtitle("Dow Jones",subtitle = "Pre-estimated model")+theme_bw()
plot2 <- GSPC_simple_backtest[[3]] + ggtitle("S&P 500",subtitle = "Pre-estimated model")+theme_bw()
plot3 <- IXIC_simple_backtest[[3]] + ggtitle("Nasdaq Composite",subtitle = "Pre-estimated model")+theme_bw()
plot4 <- N225_simple_backtest[[3]] + ggtitle("Nikkei 225",subtitle = "Pre-estimated model")+theme_bw()

plot_backtest <- grid.arrange(plot1, plot2,plot3 ,plot4 , ncol=2)

save_path <- "/home/s/Dropbox/KU/BSc Stas/R/ARIMA_final_results/preEst"
setwd(save_path)

ggsave(filename =   "DJI_returns_preest.pdf", plot = plot1, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
ggsave(filename =  "GSPC_returns_preest.pdf", plot = plot2, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
ggsave(filename =  "IXIC_returns_preest.pdf", plot = plot3, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
ggsave(filename =  "N225_returns_preest.pdf", plot = plot4, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
ggsave(filename = "Joint_returns_preest.pdf", plot = plot_backtest, device = "pdf", dpi =500, width = 8, height = 5, units = "in")

simple_backtest_output_table <- data.frame(Series = c("Dow Jones","S&P 500","Nasdaq Composite","Nikkei 225"), 
                                           RMSE= c(DJI_simple_backtest[[2]], 
                                                   GSPC_simple_backtest[[2]],
                                                   IXIC_simple_backtest[[2]],
                                                   N225_simple_backtest[[2]]))
write.table(simple_backtest_output_table,"returns_preest_table.txt",sep="\t",row.names=FALSE)
xtable(simple_backtest_output_table, digits = 8)


# Rolling window of 1 year with reestimation
### SAME MODEL:

DJI_w_rest_backtest_1yr <- one_step_w_rest(DJI_returns ,c(0,0,2),1,FALSE, "2015-03-09")
GSPC_w_rest_backtest_1yr<- one_step_w_rest(GSPC_returns,c(0,0,2),1,FALSE, "2015-03-05")
IXIC_w_rest_backtest_1yr<- one_step_w_rest(IXIC_returns,c(2,0,2),1,FALSE, "2015-03-10")
N225_w_rest_backtest_1yr<- one_step_w_rest(N225_returns,c(0,0,2),1,FALSE, "2015-02-16")


plot1 <-  DJI_w_rest_backtest_1yr[[2]] + ggtitle("Dow Jones",       subtitle = "Estimation on a rolling window of 1 year")+theme_bw()
plot2 <- GSPC_w_rest_backtest_1yr[[2]] + ggtitle("S&P 500",         subtitle = "Estimation on a rolling window of 1 year")+theme_bw()
plot3 <- IXIC_w_rest_backtest_1yr[[2]] + ggtitle("Nasdaq Composite",subtitle = "Estimation on a rolling window of 1 year")+theme_bw()
plot4 <- N225_w_rest_backtest_1yr[[2]] + ggtitle("Nikkei 225",      subtitle = "Estimation on a rolling window of 1 year")+theme_bw()

plot_backtest_w_rest_1yr <- grid.arrange(plot1, plot2,plot3 ,plot4 , ncol=2)
save_path <- "/home/s/Dropbox/KU/BSc Stas/R/ARIMA_final_results/rolling_1"
setwd(save_path)

ggsave(filename = "DJI_returns_rolling_1yr.pdf", plot = plot1, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
ggsave(filename = "GSPC_returns_rolling_1yr.pdf", plot = plot2, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
ggsave(filename = "IXIC_returns_rolling_1yr.pdf", plot = plot3, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
ggsave(filename = "N225_returns_rolling_1yr.pdf", plot = plot4, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
ggsave(filename = "Joint_returns_rolling_1yr.pdf", plot = plot_backtest_w_rest_1yr, device = "pdf", dpi =500, width = 8, height = 5, units = "in")

simple_backtest_w_rest_table_1yr <- data.frame(Series = c("Dow Jones","S&P 500","Nasdaq Composite","Nikkei 225"), 
                                           RMSE= c( DJI_w_rest_backtest_1yr[[1]], 
                                                   GSPC_w_rest_backtest_1yr[[1]],
                                                   IXIC_w_rest_backtest_1yr[[1]],
                                                   N225_w_rest_backtest_1yr[[1]])) 
xtable(simple_backtest_w_rest_table_1yr, digits = 8)
write.table(simple_backtest_w_rest_table_1yr,"returns_rolling_1yr_table.txt",sep="\t",row.names=FALSE)



# Rolling window of 3 years with reestimation
### SAME MODEL:

DJI_w_rest_backtest_3yr <- one_step_w_rest(DJI_returns ,c(0,0,2),3,FALSE,"2015-03-09")
GSPC_w_rest_backtest_3yr<- one_step_w_rest(GSPC_returns,c(0,0,2),3,FALSE,"2015-03-05")
IXIC_w_rest_backtest_3yr<- one_step_w_rest(IXIC_returns,c(2,0,2),3,FALSE,"2015-03-10")
N225_w_rest_backtest_3yr<- one_step_w_rest(N225_returns,c(0,0,2),3,FALSE,"2015-02-16")


plot1 <-  DJI_w_rest_backtest_3yr[[2]] + ggtitle("Dow Jones",       subtitle = "Estimation on a rolling window of 3 years")+theme_bw()
plot2 <- GSPC_w_rest_backtest_3yr[[2]] + ggtitle("S&P 500",         subtitle = "Estimation on a rolling window of 3 years")+theme_bw()
plot3 <- IXIC_w_rest_backtest_3yr[[2]] + ggtitle("Nasdaq Composite",subtitle = "Estimation on a rolling window of 3 years")+theme_bw()
plot4 <- N225_w_rest_backtest_3yr[[2]] + ggtitle("Nikkei 225",      subtitle = "Estimation on a rolling window of 3 years")+theme_bw()

plot_backtest_w_rest_3yr <- grid.arrange(plot1, plot2,plot3 ,plot4 , ncol=2)

save_path <- "/home/s/Dropbox/KU/BSc Stas/R/ARIMA_final_results/rolling_3"
setwd(save_path)
ggsave(filename =   "DJI_returns_rolling_3yr.pdf", plot = plot1, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
ggsave(filename =  "GSPC_returns_rolling_3yr.pdf", plot = plot2, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
ggsave(filename =  "IXIC_returns_rolling_3yr.pdf", plot = plot3, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
ggsave(filename =  "N225_returns_rolling_3yr.pdf", plot = plot4, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
ggsave(filename = "Joint_returns_rolling_3yr.pdf", plot = plot_backtest_w_rest_3yr, device = "pdf", dpi =500, width = 8, height = 5, units = "in")

simple_backtest_w_rest_table_3yr <- data.frame(Series = c("Dow Jones","S&P 500","Nasdaq Composite","Nikkei 225"), 
                                               RMSE= c(  DJI_w_rest_backtest_3yr[[1]], 
                                                        GSPC_w_rest_backtest_3yr[[1]],
                                                        IXIC_w_rest_backtest_3yr[[1]],
                                                        N225_w_rest_backtest_3yr[[1]])) 
xtable(simple_backtest_w_rest_table_3yr, digits = 8)
write.table(simple_backtest_w_rest_table_3yr,"returns_rolling_3yr_table.txt",sep="\t",row.names=FALSE)

all_results_table <- simple_backtest_output_table %>% mutate(without_reestimation = RMSE) %>% select(-RMSE) %>% 
  left_join(simple_backtest_w_rest_table_1yr %>% mutate(rolling_1yr = RMSE) %>% select(-RMSE),by = "Series") %>% 
  left_join(simple_backtest_w_rest_table_3yr %>% mutate(rolling_3yr = RMSE) %>% select(-RMSE),by = "Series")
write.table(all_results_table,"all_results_returns.txt",sep="\t",row.names=FALSE)
xtable(all_results_table, digits = 8)


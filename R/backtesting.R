### Backtesting m8

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

DJI_csv <- as.xts(read.csv.zoo("DJI.csv"))
GSPC_csv <- as.xts(read.csv.zoo("GSPC.csv"))
IXIC_csv <- as.xts(read.csv.zoo("IXIC.csv"))
N225_csv <- as.xts(read.csv.zoo("N225.csv"))

storage.mode(N225_csv) <- "numeric"
N225_csv <- na.locf(N225_csv)

GSPC_log <- log(GSPC_csv)
DJI_log <- log(DJI_csv)
IXIC_log <- log(IXIC_csv)
N225_log <- log(N225_csv)


GSPC_returns<- diff(GSPC_log$Adj.Close) %>% na.approx()
DJI_returns <-  diff(DJI_log$Adj.Close) %>% na.approx()
IXIC_returns <-  diff(IXIC_log$Adj.Close) %>% na.approx()
N225_returns <-  diff(N225_log$Adj.Close) %>% na.approx()



one_step_no_rest <- function(dataset, arma, split,label){
  train <- xts::first(dataset, as.integer(length(dataset)*split))
  test <- xts::last(dataset, length(dataset)- as.integer(length(dataset)*split))
  model <- auto.arima(train, max.d = arma[1], max.p = arma[2], max.q = arma[3])
  
  
  test_set_model_arma <- Arima(test, model=model)
  predictions_dt <-timeSeries(test_set_model_arma$fitted,index(test)) %>% as.xts()
  
  plot <- ggplot() + 
    geom_line(data = test, aes(x = index(test), y = test, color = "red"),size =1) +
    geom_line(data = predictions_dt, aes(x = index(test), y = predictions_dt, color = "darkblue"),size = 1) +
    xlab('Time') +
    ylab('Retunrs')+scale_color_discrete(name = "Series", labels = c("Predicted","Actual"))
  print(plot)
  ggsave(filename = paste0(label,"_done_right.pdf"), plot = plot, device ="pdf", dpi = 300)
  
  summary(model)
  
  manual_rmse <- Metrics::rmse(as.numeric(test$Adj.Close),as.numeric(predictions_dt$TS.1))
  
  automatic_rmse_train <- accuracy(model)[2]
  print(automatic_rmse_train)
  automatic_rmse <- accuracy(test_set_model_arma)[2]
  output <- list(manual_rmse,automatic_rmse, plot)
  return(output)
}



check_if_weekend <- function(date){
  if(wday(date,label = FALSE) %in% c(6,7)){
    print(wday(date,label = FALSE,week_start=1))
    return(TRUE)
  }
  else
    print(wday(date,label = FALSE,week_start=1))
    return(FALSE)
}

one_step_w_rest <- function(dataset, arma, window_length,auto,label){
  dataset <- dataset[index(dataset)>="2010-01-01"]
  start = as.Date(index(dataset)[1])
  end = index(dataset)[253*window_length]
  end_of_loop <- length(dataset)-253*window_length
  test <- xts::last(dataset, end_of_loop-1)
  
  preds <- c()
  label <- c()
  i = 1
  while(i < end_of_loop){
    #print(i)
    train <- window(dataset,  start = start, end = end)
    
    if(auto){
      model <- auto.arima(train)
    }
    else{
      model <- Arima(train, order = arma)
      }
    preds[i] <- forecast::forecast(model)$mean[1]
    label[i] <- dataset[253*window_length+i]
    
    # NEXT:
    i = i+1
    start = index(dataset)[i]
    end = index(dataset)[253*window_length+i]
  }
  
  loss <- Metrics::rmse(label,preds)
  
  predictions_dt <-data.frame(predict =preds) 
  

  plot <- ggplot() + 
    geom_line(data = test, aes(x = index(test), y = test, color = "red"),size =1) +
    geom_line(data = predictions_dt, aes(x = index(test), y = predictions_dt$predict, color = "darkblue"),size = 1) +
    xlab('Time') +
    ylab('Retunrs')+scale_color_discrete(name = "Series", labels = c("Predicted","Actual"))
  #print(plot)
  ggsave(filename = paste0(label,"_wrest.pdf"), plot = plot, device ="pdf", dpi = 300)
  output <- list(loss,plot)
  return(output)
}

DJI_simple_backtest <- one_step_no_rest(DJI_returns ,c(0,0,2),0.7,"DJI_preds_vs_actual" )
GSPC_simple_backtest<- one_step_no_rest(GSPC_returns,c(0,0,2),0.7,"GSPC_preds_vs_actual")
IXIC_simple_backtest<- one_step_no_rest(IXIC_returns,c(0,0,2),0.7,"IXIC_preds_vs_actual")
N225_simple_backtest<- one_step_no_rest(N225_returns,c(0,0,2),0.7,"N225_preds_vs_actual")


plot1 <- DJI_simple_backtest[[3]]  + ggtitle("Dow Jones",subtitle = "One-step forecast without re-estimation")+theme_bw()
plot2 <- GSPC_simple_backtest[[3]] + ggtitle("S&P 500",subtitle = "One-step forecast without re-estimation")+theme_bw()
plot3 <- IXIC_simple_backtest[[3]] + ggtitle("Nasdaq Composite",subtitle = "One-step forecast without re-estimation")+theme_bw()
plot4 <- N225_simple_backtest[[3]] + ggtitle("Nikkei 225",subtitle = "One-step forecast without re-estimation")+theme_bw()
require(gridExtra)

plot_backtest <- grid.arrange(plot1, plot2,plot3 ,plot4 , ncol=2)
ggsave(filename = "plot1_backtest_simple.pdf", plot = plot1, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
ggsave(filename = "plot2_backtest_simple.pdf", plot = plot2, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
ggsave(filename = "plot3_backtest_simple.pdf", plot = plot3, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
ggsave(filename = "plot4_backtest_simple.pdf", plot = plot4, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")

simple_backtest_output_table <- data.frame(Series = c("Dow Jones","S&P 500","Nasdaq Composite","Nikkei 225"), 
                                           RMSE= c(DJI_simple_backtest[[2]], 
                                                   GSPC_simple_backtest[[2]],
                                                   IXIC_simple_backtest[[2]],
                                                   N225_simple_backtest[[2]])) 
xtable(simple_backtest_output_table, digits = 8)


DJI_w_rest_backtest_5yr <- one_step_w_rest(DJI_returns ,c(0,0,2),5,FALSE,"DJI_preds_vs_actual" )
GSPC_w_rest_backtest_5yr<- one_step_w_rest(GSPC_returns,c(0,0,2),5,FALSE,"GSPC_preds_vs_actual")
IXIC_w_rest_backtest_5yr<- one_step_w_rest(IXIC_returns,c(0,0,2),5,FALSE,"IXIC_preds_vs_actual")
N225_w_rest_backtest_5yr<- one_step_w_rest(N225_returns,c(0,0,2),5,FALSE,"N225_preds_vs_actual")


plot1 <-  DJI_w_rest_backtest_5yr[[2]] + ggtitle("Dow Jones",       subtitle = "Rolling window forecast with re-estimation")+theme_bw()
plot2 <- GSPC_w_rest_backtest_5yr[[2]] + ggtitle("S&P 500",         subtitle = "Rolling window forecast with re-estimation")+theme_bw()
plot3 <- IXIC_w_rest_backtest_5yr[[2]] + ggtitle("Nasdaq Composite",subtitle = "Rolling window forecast with re-estimation")+theme_bw()
plot4 <- N225_w_rest_backtest_5yr[[2]] + ggtitle("Nikkei 225",      subtitle = "Rolling window forecast with re-estimation")+theme_bw()
require(gridExtra)

plot_backtest_w_rest <- grid.arrange(plot1, plot2,plot3 ,plot4 , ncol=2)
ggsave(filename = "plot1_w_rest_backtest_5yr.pdf", plot = plot1, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
ggsave(filename = "plot2_w_rest_backtest_5yr.pdf", plot = plot2, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
ggsave(filename = "plot3_w_rest_backtest_5yr.pdf", plot = plot3, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
ggsave(filename = "plot4_w_rest_backtest_5yr.pdf", plot = plot4, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")

simple_backtest_w_rest_table_5yr <- data.frame(Series = c("Dow Jones","S&P 500","Nasdaq Composite","Nikkei 225"), 
                                           RMSE= c( DJI_w_rest_backtest_5yr[[1]], 
                                                   GSPC_w_rest_backtest_5yr[[1]],
                                                   IXIC_w_rest_backtest_5yr[[1]],
                                                   N225_w_rest_backtest_5yr[[1]])) 
xtable(simple_backtest_w_rest_table_5yr, digits = 8)


DJI_w_rest_backtest_auto_5yr <- one_step_w_rest(DJI_returns ,c(0,0,2),5,TRUE,"DJI_preds_vs_actual_auto" )
GSPC_w_rest_backtest_auto_5yr<- one_step_w_rest(GSPC_returns,c(0,0,2),5,TRUE,"GSPC_preds_vs_actual_auto")
IXIC_w_rest_backtest_auto_5yr<- one_step_w_rest(IXIC_returns,c(0,0,2),5,TRUE,"IXIC_preds_vs_actual_auto")
N225_w_rest_backtest_auto_5yr<- one_step_w_rest(N225_returns,c(0,0,2),5,TRUE,"N225_preds_vs_actual_auto")



plot1 <-  DJI_w_rest_backtest_auto_5yr[[2]]  + ggtitle("Dow Jones",       subtitle = "Rolling window forecast with re-estimation")+theme_bw()
plot2 <- GSPC_w_rest_backtest_auto_5yr[[2]] + ggtitle("S&P 500",         subtitle = "Rolling window forecast with re-estimation")+theme_bw()
plot3 <- IXIC_w_rest_backtest_auto_5yr[[2]] + ggtitle("Nasdaq Composite",subtitle = "Rolling window forecast with re-estimation")+theme_bw()
plot4 <- N225_w_rest_backtest_auto_5yr[[2]] + ggtitle("Nikkei 225",      subtitle = "Rolling window forecast with re-estimation")+theme_bw()
require(gridExtra)

plot_backtest_w_rest <- grid.arrange(plot1, plot2,plot3 ,plot4 , ncol=2)
ggsave(filename = "plot1_w_rest_backtest_auto_5yr.pdf", plot = plot1, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
ggsave(filename = "plot2_w_rest_backtest_auto_5yr.pdf", plot = plot2, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
ggsave(filename = "plot3_w_rest_backtest_auto_5yr.pdf", plot = plot3, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
ggsave(filename = "plot4_w_rest_backtest_auto_5yr.pdf", plot = plot4, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")

simple_backtest_w_rest_auto_table_5yr <- data.frame(Series = c("Dow Jones","S&P 500","Nasdaq Composite","Nikkei 225"), 
                                           RMSE= c(  DJI_w_rest_backtest_auto_5yr[[1]], 
                                                    GSPC_w_rest_backtest_auto_5yr[[1]],
                                                    IXIC_w_rest_backtest_auto_5yr[[1]],
                                                    N225_w_rest_backtest_auto_5yr[[1]])) 
xtable(simple_backtest_w_rest_auto_table_5yr, digits = 8)


cumsum_resutl <- simple_backtest_output_table %>% mutate(without_reestimation = RMSE) %>% select(-RMSE) %>% left_join(simple_backtest_w_rest_table %>% mutate(with_reestimation = RMSE) %>% select(-RMSE),by = "Series") %>% 
  left_join(simple_backtest_w_rest_table_3yr %>% mutate(with_reestimation_3yr = RMSE) %>% select(-RMSE),by = "Series") %>% 
  left_join(simple_backtest_w_rest_table_5yr %>% mutate(with_reestimation__5yr = RMSE) %>% select(-RMSE),by = "Series") %>% 
  left_join(simple_backtest_w_rest_auto_table %>% mutate(with_reestimation_auto = RMSE) %>% select(-RMSE),by = "Series") %>%
  left_join(simple_backtest_w_rest_auto_table_3yr %>% mutate(with_reestimation_auto_3yr = RMSE) %>% select(-RMSE),by = "Series") %>% 
  left_join(simple_backtest_w_rest_auto_table_5yr %>% mutate(with_reestimation_auto_5yr = RMSE) %>% select(-RMSE),by = "Series")


xtable(cumsum_resutl, digits = 8)

results <- list(
   DJI_simple_backtest = DJI_simple_backtest 
  ,GSPC_simple_backtest= GSPC_simple_backtest
  ,IXIC_simple_backtest= IXIC_simple_backtest
  ,N225_simple_backtest= N225_simple_backtest
  , DJI_w_rest_backtest = DJI_w_rest_backtest  
  ,GSPC_w_rest_backtest= GSPC_w_rest_backtest
  ,IXIC_w_rest_backtest= IXIC_w_rest_backtest
  ,N225_w_rest_backtest= N225_w_rest_backtest
  , DJI_w_rest_backtest_3yr = DJI_w_rest_backtest_3yr 
  ,GSPC_w_rest_backtest_3yr= GSPC_w_rest_backtest_3yr
  ,IXIC_w_rest_backtest_3yr= IXIC_w_rest_backtest_3yr
  ,N225_w_rest_backtest_3yr= N225_w_rest_backtest_3yr
  , DJI_w_rest_backtest_5yr = DJI_w_rest_backtest_5yr 
  ,GSPC_w_rest_backtest_5yr= GSPC_w_rest_backtest_5yr
  ,IXIC_w_rest_backtest_5yr= IXIC_w_rest_backtest_5yr
  ,N225_w_rest_backtest_5yr= N225_w_rest_backtest_5yr
  , DJI_w_rest_backtest_auto_3yr = DJI_w_rest_backtest_auto_3yr 
  ,GSPC_w_rest_backtest_auto_3yr= GSPC_w_rest_backtest_auto_3yr
  ,IXIC_w_rest_backtest_auto_3yr= IXIC_w_rest_backtest_auto_3yr
  ,N225_w_rest_backtest_auto_3yr= N225_w_rest_backtest_auto_3yr
  , DJI_w_rest_backtest_auto_5yr = DJI_w_rest_backtest_auto_5yr 
  ,GSPC_w_rest_backtest_auto_5yr= GSPC_w_rest_backtest_auto_5yr
  ,IXIC_w_rest_backtest_auto_5yr= IXIC_w_rest_backtest_auto_5yr
  ,N225_w_rest_backtest_auto_5yr= N225_w_rest_backtest_auto_5yr,
  result_table = cumsum_resutl
)

save(results, file = "results.RData")



### Backtesting volatility
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
dt <- read.csv("data/data_volatility/oxfordmanrealizedvolatilityindices.csv",stringsAsFactors = F)
colnames(dt)[1] <- "Datetime"
dt$Datetime <- as.Date(dt$Datetime)

unique(dt$Symbol)

DJI_volatility <- dt %>% filter(Symbol == ".DJI") 

model_dji <- auto.arima(DJI_volatility$rv5[DJI_volatility$Datetime < "2017-01-01"], stepwise = FALSE, method = "ML")

GSPC_volatility <- dt %>% filter(Symbol == ".SPX") 

model_spx <- auto.arima(GSPC_volatility$rv5[GSPC_volatility$Datetime < "2017-01-01"], stepwise = FALSE, method = "ML")

summary(model_dji)
summary(model_spx)



# Backtesting functions:
one_step_no_rest <- function(dataset, start){
  train <- dataset %>% filter(Datetime < start)
  test <-  dataset %>% filter(Datetime >= start)
  print(head(test))
  model <- auto.arima(train$rv5, method = "ML", stepwise = FALSE)
  summary(model)
  test_set_model_arma <- Arima(test$rv5, model=model)
  predictions_dt <-test_set_model_arma$fitted
  
  plot_df <- test %>% mutate(PREDICTIONS = as.numeric(predictions_dt)) %>% head(-1)
  print(dim(plot_df))
  plot <- ggplot(plot_df, aes(x = Datetime)) + 
    geom_line(aes( y = rv5, color = "red"),size = 0.5) +
    geom_line(aes( y = PREDICTIONS , color = "darkblue"),size = 0.5) +
    xlab('Time') +
    ylab('Realised Volatility')+scale_color_discrete(name = "Series", labels = c("Predicted","Actual"))+theme_bw()
  manual_rmse <- Metrics::rmse(as.numeric(plot_df$rv5),as.numeric(plot_df$PREDICTIONS))
  
  
  automatic_rmse_train <- accuracy(model)[2]
  automatic_rmse <- accuracy(test_set_model_arma)[2]
  output <- list(automatic_rmse,manual_rmse, plot)
  return(output)
}
one_step_w_rest <- function(dataset, arma, window_length, auto, eval_start){
  
  if (window_length == 3){
    dataset <- dataset[dataset$Datetime >="2015-01-01",]
  }
  if (window_length == 1){
    dataset <- dataset[dataset$Datetime >="2017-01-01",]
  }
  
  
  end = 253*window_length
  
  end_of_loop <- dim(dataset)[1]-253*window_length+1
  test <- tail(dataset,end_of_loop)
  print(length(test))
  preds <- c()
  labels <- dataset[dataset$Datetime >= dataset$Datetime[end+1],]
  i = 1
  while(i < end_of_loop){
    #print(i)
    train <- dataset$rv5[i:end]
    
    if(auto){
      model <- auto.arima(train, stepwise = FALSE, method = "ML")
    }
    else{
      model <- Arima(train, order = arma, method ="ML")
    }
    preds[i] <- forecast::forecast(model)$mean[1]
    
    # NEXT:
    i = i+1
    end = end+1
  }
  print("got to here")
  print(length(labels))
  print(length(preds))
  
  
  plot_dt <- labels %>% mutate(PREDICTIONS = preds) %>% filter(Datetime >=eval_start) %>% head(-1)
  print( "DIMENSIONS AND FIRST DATE")
  print(dim(plot_dt))
  print(plot_dt$Datetime[1])
  print( "DIMENSIONS AND FIRST DATE")
  loss <- Metrics::rmse(plot_dt$rv5,plot_dt$PREDICTIONS)
  print(loss)
  plot <- ggplot(plot_dt, aes(x = Datetime)) + 
   geom_line(aes( y = rv5, color = "red"), size =0.5) +
   geom_line(aes( y = PREDICTIONS, color = "darkblue"),size = 0.5) +
    xlab('Time') +
    ylab('Realised Volatility')+scale_color_discrete(name = "Series", labels = c("Predicted","Actual"))+theme_bw()
  
  output <- list(loss,plot)
  return(output)
}


save_path <- "/home/s/Dropbox/KU/BSc Stas/R/ARIMA_final_results/preEst"
setwd(save_path)

# One-step without reestimation: 
DJI_simple_backtest <- one_step_no_rest(DJI_volatility, "2018-03-01")
GSPC_simple_backtest<- one_step_no_rest(GSPC_volatility, "2018-02-15")



plot1 <- DJI_simple_backtest[[3]]  + ggtitle("Dow Jones",subtitle = "Pre-estimated model")+theme_bw()
plot2 <- GSPC_simple_backtest[[3]] + ggtitle("S&P 500",subtitle = "Pre-estimated model")+theme_bw()

plot_backtest <- grid.arrange(plot1, plot2, ncol=2)
ggsave(filename = "DJI_volatility_preest.pdf", plot = plot1, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
ggsave(filename = "GSPC_volatility_preest.pdf", plot = plot2, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
ggsave(filename = "Joint_volatility_preest.pdf", plot = plot_backtest, device = "pdf", dpi =500, width = 8, height = 5, units = "in")


simple_backtest_output_table <- data.frame(Series = c("Dow Jones","S&P 500"), 
                                           RMSE= c(DJI_simple_backtest[[2]], 
                                                   GSPC_simple_backtest[[2]]
                                                   ))
write.table(simple_backtest_output_table,"volatility_preest_table.txt",sep="\t",row.names=FALSE)
xtable(simple_backtest_output_table, digits = 8)


# Rolling window of 1 year with reestimation
### SAME MODEL:

save_path <- "/home/s/Dropbox/KU/BSc Stas/R/ARIMA_final_results/rolling_1"
setwd(save_path)

DJI_w_rest_backtest_1yr <- one_step_w_rest(DJI_volatility ,c(4,1,1),1,FALSE, "2018-03-01")
GSPC_w_rest_backtest_1yr<- one_step_w_rest(GSPC_volatility,c(4,1,1),1,FALSE, "2018-02-15")


plot1 <-  DJI_w_rest_backtest_1yr[[2]] + ggtitle("Dow Jones",       subtitle = "Estimation on a rolling window of 1 year")+theme_bw()
plot2 <- GSPC_w_rest_backtest_1yr[[2]] + ggtitle("S&P 500",         subtitle = "Estimation on a rolling window of 1 year")+theme_bw()

plot_backtest_w_rest_1yr <- grid.arrange(plot1, plot2, ncol=2)
ggsave(filename = "DJI_volatility_rolling_1yr.pdf", plot = plot1, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
ggsave(filename = "GSPC_volatility_rolling_1yr.pdf", plot = plot2, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
ggsave(filename = "Joint_volatility_rolling_1yr.pdf", plot = plot_backtest_w_rest_1yr, device = "pdf", dpi =500, width = 8, height = 5, units = "in")

simple_backtest_w_rest_table_1yr <- data.frame(Series = c("Dow Jones","S&P 500"), 
                                               RMSE= c( DJI_w_rest_backtest_1yr[[1]], 
                                                        GSPC_w_rest_backtest_1yr[[1]]
                                                        )) 
xtable(simple_backtest_w_rest_table_1yr, digits = 8)
write.table(simple_backtest_w_rest_table_1yr,"volatility_rolling_1.txt",sep="\t",row.names=FALSE)



# Rolling window of 3 years with reestimation
### SAME MODEL:
save_path <- "/home/s/Dropbox/KU/BSc Stas/R/ARIMA_final_results/rolling_3"
setwd(save_path)
DJI_w_rest_backtest_3yr <- one_step_w_rest(DJI_volatility ,c(0,1,5),3,FALSE,"2018-03-01")
GSPC_w_rest_backtest_3yr<- one_step_w_rest(GSPC_volatility,c(0,1,5),3,FALSE,"2018-02-15")



plot1 <-  DJI_w_rest_backtest_3yr[[2]] + ggtitle("Dow Jones",       subtitle = "Estimation on a rolling window of 3 years")+theme_bw()
plot2 <- GSPC_w_rest_backtest_3yr[[2]] + ggtitle("S&P 500",         subtitle = "Estimation on a rolling window of 3 years")+theme_bw()

plot_backtest_w_rest_3yr <- grid.arrange(plot1, plot2,ncol=2)
ggsave(filename =   "DJI_volatility_rolling_3yr.pdf", plot = plot1, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
ggsave(filename =  "GSPC_volatility_rolling_3yr.pdf", plot = plot2, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
ggsave(filename = "Joint_volatility_rolling_3yr.pdf", plot = plot_backtest_w_rest_3yr, device = "pdf", dpi =500, width = 8, height = 5, units = "in")
    
simple_backtest_w_rest_table_3yr <- data.frame(Series = c("Dow Jones","S&P 500"), 
                                               RMSE= c(  DJI_w_rest_backtest_3yr[[1]], 
                                                         GSPC_w_rest_backtest_3yr[[1]]
                                                         ) )
xtable(simple_backtest_w_rest_table_3yr, digits = 8)
write.table(simple_backtest_w_rest_table_3yr,"volatility_rolling_3yr.txt",sep="\t",row.names=FALSE)


all_results_table <- simple_backtest_output_table %>% mutate(without_reestimation = RMSE) %>% select(-RMSE) %>% 
  left_join(simple_backtest_w_rest_table_1yr %>% mutate(rolling_1yr = RMSE) %>% select(-RMSE),by = "Series") %>% 
  left_join(simple_backtest_w_rest_table_3yr %>% mutate(rolling_3yr = RMSE) %>% select(-RMSE),by = "Series")
write.table(all_results_table,"all_results_volatility.txt",sep="\t",row.names=FALSE)
xtable(all_results_table, digits = 8)


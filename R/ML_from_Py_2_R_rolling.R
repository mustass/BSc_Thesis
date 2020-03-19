rm(list = ls())
library(reticulate)
library(ggplot2)
library(gridExtra)
library(xtable)
library(dplyr)
library(xts)
use_condaenv(condaenv = "skorch")
py_run_string("sys.path.append('/home/s/Dropbox/KU/BSc Stas/Python/Try_again/')")

source_python('/home/s/Dropbox/KU/BSc Stas/Python/Try_again/utils/rolling_window_results.py')
DJI_rolling_1_returns_result <- get_rolling_window_results( 'returns','rolling_1',"DJI")
GSPC_rolling_1_returns_result <- get_rolling_window_results('returns','rolling_1',"GSPC")
N225_rolling_1_returns_result <- get_rolling_window_results('returns','rolling_1',"N225")
IXIC_rolling_1_returns_result <- get_rolling_window_results('returns','rolling_1',"IXIC")

DJI_rolling_3_returns_result <- get_rolling_window_results( 'returns','rolling_3',"DJI")
GSPC_rolling_3_returns_result <- get_rolling_window_results('returns','rolling_3',"GSPC")
N225_rolling_3_returns_result <- get_rolling_window_results('returns','rolling_3',"N225")
IXIC_rolling_3_returns_result <- get_rolling_window_results('returns','rolling_3',"IXIC")

DJI_rolling_1_volatility_result <- get_rolling_window_results('volatility','rolling_1',"DJI")
GSPC_rolling_1_volatility_result <- get_rolling_window_results('volatility','rolling_1',"GSPC")
DJI_rolling_3_volatility_result <- get_rolling_window_results('volatility','rolling_3',"DJI")
GSPC_rolling_3_volatility_result <- get_rolling_window_results('volatility','rolling_3',"GSPC")

dt <- read.csv("/home/s/Dropbox/KU/BSc Stas/R/data/data_volatility/oxfordmanrealizedvolatilityindices.csv",stringsAsFactors = F)
colnames(dt)[1] <- "Datetime"
dt$Datetime <- as.Date(dt$Datetime)

unique(dt$Symbol)

DJI_volatility <- dt %>% filter(Symbol == ".DJI") 
GSPC_volatility <- dt %>% filter(Symbol == ".SPX") 

DJI_csv <-  read.csv.zoo("/home/s/Dropbox/KU/BSc Stas/R/data/data_returns/DJI.csv") %>% as.data.frame() %>%  mutate(Datetime = rownames(read.csv.zoo("/home/s/Dropbox/KU/BSc Stas/R/data/data_returns/DJI.csv")%>% as.data.frame()))
GSPC_csv <- read.csv.zoo("/home/s/Dropbox/KU/BSc Stas/R/data/data_returns/GSPC.csv") %>% as.data.frame() %>%  mutate(Datetime = rownames(read.csv.zoo("/home/s/Dropbox/KU/BSc Stas/R/data/data_returns/GSPC.csv")%>% as.data.frame()))
IXIC_csv <- read.csv.zoo("/home/s/Dropbox/KU/BSc Stas/R/data/data_returns/IXIC.csv") %>% as.data.frame() %>%  mutate(Datetime = rownames(read.csv.zoo("/home/s/Dropbox/KU/BSc Stas/R/data/data_returns/IXIC.csv")%>% as.data.frame()))
N225_csv <- read.csv.zoo("/home/s/Dropbox/KU/BSc Stas/R/data/data_returns/N225.csv") %>% as.data.frame() %>%  mutate(Datetime = rownames(read.csv.zoo("/home/s/Dropbox/KU/BSc Stas/R/data/data_returns/N225.csv")%>% as.data.frame()))



# FIND WHERE THEY OVERLAP:

DJI_rolling_1_returns_result[[1]][DJI_rolling_3_returns_result[[1]]$true_vals[1] == DJI_rolling_1_returns_result[[1]]$true_vals,]
DJI_returns_2015_forecast_1yrs <- DJI_rolling_1_returns_result[[1]][507:dim(DJI_rolling_1_returns_result[[1]])[1],]

GSPC_rolling_1_returns_result[[1]][GSPC_rolling_3_returns_result[[1]]$true_vals[1] == GSPC_rolling_1_returns_result[[1]]$true_vals,]
GSPC_returns_2015_forecast_1yrs <- GSPC_rolling_1_returns_result[[1]][507:dim(GSPC_rolling_1_returns_result[[1]])[1],]

IXIC_rolling_1_returns_result[[1]][IXIC_rolling_3_returns_result[[1]]$true_vals[1] == IXIC_rolling_1_returns_result[[1]]$true_vals,]
IXIC_returns_2015_forecast_1yrs <- IXIC_rolling_1_returns_result[[1]][507:dim(IXIC_rolling_1_returns_result[[1]])[1],]

N225_rolling_1_returns_result[[1]][N225_rolling_3_returns_result[[1]]$true_vals[1] == N225_rolling_1_returns_result[[1]]$true_vals,]
N225_returns_2015_forecast_1yrs <- N225_rolling_1_returns_result[[1]][507:dim(N225_rolling_1_returns_result[[1]])[1],]

DJI_rolling_1_volatility_result[[1]][DJI_rolling_3_volatility_result[[1]]$true_vals[1] == DJI_rolling_1_volatility_result[[1]]$true_vals,]
DJI_volatilit_2015_forecast_1yrs <- DJI_rolling_1_volatility_result[[1]][507:dim(DJI_rolling_1_volatility_result[[1]])[1],]

GSPC_rolling_1_volatility_result[[1]][GSPC_rolling_3_volatility_result[[1]]$true_vals[1] == GSPC_rolling_1_volatility_result[[1]]$true_vals,]
GSPC_volatilit_2015_forecast_1yrs <- GSPC_rolling_1_volatility_result[[1]][507:dim(GSPC_rolling_1_volatility_result[[1]])[1],]



DJI_returns_dates  <- DJI_csv %>% filter(Datetime >= "2015-03-09") %>% select(Datetime) %>% head(-1)
DJI_returns_dates <- DJI_returns_dates$Datetime %>% as.Date()
GSPC_returns_dates <- GSPC_csv %>% filter(Datetime >= "2015-03-05") %>% select(Datetime) %>% head(-1)
GSPC_returns_dates <- GSPC_returns_dates$Datetime %>% as.Date()
N225_returns_dates <- N225_csv %>% filter(Datetime >= "2015-02-16") %>% select(Datetime) %>% head(-1)
N225_returns_dates <- N225_returns_dates$Datetime %>% as.Date()
IXIC_returns_dates <- IXIC_csv %>% filter(Datetime >= "2015-03-10") %>% select(Datetime) %>% head(-1)
IXIC_returns_dates <- IXIC_returns_dates$Datetime %>% as.Date()

DJI_volatility_dates  <- DJI_volatility %>% filter(Datetime >= "2018-03-01") %>% select(Datetime) %>% head(-1)
DJI_volatility_dates <- DJI_volatility_dates$Datetime %>% as.Date()
GSPC_volatility_dates <- GSPC_volatility %>% filter(Datetime >= "2018-02-15") %>% select(Datetime) %>% head(-1)
GSPC_volatility_dates <- GSPC_volatility_dates$Datetime %>% as.Date()


# RETURNS ROLLING 1


 DJI_returns_df <- DJI_returns_2015_forecast_1yrs %>% mutate(Date= DJI_returns_dates)
GSPC_returns_df <- GSPC_returns_2015_forecast_1yrs %>% mutate(Date= GSPC_returns_dates) 
IXIC_returns_df <- IXIC_returns_2015_forecast_1yrs %>% mutate(Date= IXIC_returns_dates) 
N225_returns_df <- N225_returns_2015_forecast_1yrs %>% mutate(Date= N225_returns_dates)


DJI_plot <- ggplot2::ggplot(DJI_returns_df,aes(x = Date)) + 
  geom_line(aes(y = true_vals, color = "red",group = 1),size =0.5) +
  geom_line(aes(y = preds, color = "darkblue",group = 1),size = 0.5) +
  xlab('Time') +
  ylab('Log-returns')+scale_color_discrete(name = "Series", labels = c("Predicted","Actual"))+
  ggtitle("Dow Jones",subtitle = "Estimation on a rolling window of 1 year")+theme_bw()

GSPC_plot <- ggplot2::ggplot() + 
  geom_line(data =GSPC_returns_df , aes(x = Date, y = true_vals, color = "red",group = 1),size =0.5) +
  geom_line(data = GSPC_returns_df, aes(x = Date, y = preds, color = "darkblue",group = 1),size =0.5) +
  xlab('Time') +
  ylab('Log-returns')+scale_color_discrete(name = "Series", labels = c("Predicted","Actual"))+
  ggtitle("S&P500",subtitle = "Estimation on a rolling window of 1 year")+theme_bw()

IXIC_plot <- ggplot2::ggplot() + 
  geom_line(data = IXIC_returns_df, aes(x = Date, y = true_vals, color = "red",group = 1),size =0.5) +
  geom_line(data = IXIC_returns_df, aes(x = Date, y = preds, color = "darkblue",group = 1),size =0.5) +
  xlab('Time') +
  ylab('Log-returns')+scale_color_discrete(name = "Series", labels = c("Predicted","Actual"))+
  ggtitle("Nasdaq Composite",subtitle = "Estimation on a rolling window of 1 year")+theme_bw()

N225_plot <- ggplot2::ggplot() + 
  geom_line(data = N225_returns_df, aes(x = Date, y = true_vals, color = "red",group = 1),size =0.5) +
  geom_line(data = N225_returns_df, aes(x = Date, y = preds, color = "darkblue",group = 1),size =0.5) +
  xlab('Time') +
  ylab('Log-returns')+scale_color_discrete(name = "Series", labels = c("Predicted","Actual"))+
  ggtitle("Nikkei 225",subtitle = "Estimation on a rolling window of 1 year")+theme_bw()

plot_backtest <- grid.arrange(DJI_plot,
                              GSPC_plot,
                              IXIC_plot,
                              N225_plot , ncol=2)
setwd("/home/s/Dropbox/KU/BSc Stas/R/ML_final_results/rolling_window_1")
ggsave(filename = "DJI_returns_rolling_1_ML.pdf", plot = DJI_plot, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
ggsave(filename = "GSPC_returns_rolling_1_ML.pdf", plot = GSPC_plot, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
ggsave(filename = "IXIC_returns_rolling_1_ML.pdf", plot = IXIC_plot, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
ggsave(filename = "N225_returns_rolling_1_ML.pdf", plot = N225_plot, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
ggsave(filename = "Joint_returns_rolling_1_ML.pdf", plot = plot_backtest, device = "pdf", dpi =500, width = 8, height = 5, units = "in")

dji_rmse_returns_roll_1  <- Metrics::rmse(DJI_returns_df$true_vals,DJI_returns_df$preds)
gspc_rmse_returns_roll_1   <- Metrics::rmse(GSPC_returns_df$true_vals,GSPC_returns_df$preds)
ixic_rmse_returns_roll_1  <- Metrics::rmse(IXIC_returns_df$true_vals,IXIC_returns_df$preds)
n225_rmse_returns_roll_1   <- Metrics::rmse(N225_returns_df$true_vals,N225_returns_df$preds)


rolling_1_returns_output_table<- data.frame(Series = c("Dow Jones","S&P 500","Nasdaq Composite","Nikkei 225"), 
                                           RMSE= c(dji_rmse_returns_roll_1, 
                                                   gspc_rmse_returns_roll_1,
                                                   ixic_rmse_returns_roll_1,
                                                   n225_rmse_returns_roll_1))
write.table(rolling_1_returns_output_table,"rolling_1_returns_output_table.txt",sep="\t",row.names=FALSE)
xtable(rolling_1_returns_output_table, digits = 8)


# RETURNS ROLLING 3

DJI_returns_df <-  DJI_rolling_3_returns_result[[1]] %>% mutate(Date= DJI_returns_dates)
GSPC_returns_df <- GSPC_rolling_3_returns_result[[1]] %>% mutate(Date= GSPC_returns_dates) 
IXIC_returns_df <- IXIC_rolling_3_returns_result[[1]] %>% mutate(Date= IXIC_returns_dates) 
N225_returns_df <- N225_rolling_3_returns_result[[1]] %>% mutate(Date= N225_returns_dates)


DJI_plot <- ggplot2::ggplot() + 
  geom_line(data = DJI_returns_df, aes(x = Date, y = true_vals, color = "red",group = 1),size =0.5) +
  geom_line(data = DJI_returns_df, aes(x = Date, y = preds, color = "darkblue",group = 1),size =0.5) +
  xlab('Time') +
  ylab('Log-returns')+scale_color_discrete(name = "Series", labels = c("Predicted","Actual"))+
  ggtitle("Dow Jones",subtitle = "Estimation on a rolling window of 3 years")+theme_bw()

GSPC_plot <- ggplot2::ggplot() + 
  geom_line(data = GSPC_returns_df, aes(x = Date, y = true_vals, color = "red",group = 1),size =0.5) +
  geom_line(data = GSPC_returns_df, aes(x = Date, y = preds, color = "darkblue",group = 1),size =0.5) +
  xlab('Time') +
  ylab('Log-returns')+scale_color_discrete(name = "Series", labels = c("Predicted","Actual"))+
  ggtitle("S&P500",subtitle = "Estimation on a rolling window of 3 years")+theme_bw()

IXIC_plot <- ggplot2::ggplot() + 
  geom_line(data = IXIC_returns_df, aes(x = Date, y = true_vals, color = "red",group = 1),size =0.5) +
  geom_line(data = IXIC_returns_df, aes(x = Date, y = preds, color = "darkblue",group = 1),size =0.5) +
  xlab('Time') +
  ylab('Log-returns')+scale_color_discrete(name = "Series", labels = c("Predicted","Actual"))+
  ggtitle("Nasdaq Composite",subtitle = "Estimation on a rolling window of 3 years")+theme_bw()

N225_plot <- ggplot2::ggplot() + 
  geom_line(data = N225_returns_df, aes(x = Date, y = true_vals, color = "red",group = 1),size =0.5) +
  geom_line(data = N225_returns_df, aes(x = Date, y = preds, color = "darkblue",group = 1),size =0.5) +
  xlab('Time') +
  ylab('Log-returns')+scale_color_discrete(name = "Series", labels = c("Predicted","Actual"))+
  ggtitle("Nikkei 225",subtitle = "Estimation on a rolling window of 3 years")+theme_bw()

plot_backtest <- grid.arrange(DJI_plot,
                              GSPC_plot,
                              IXIC_plot,
                              N225_plot , ncol=2)
setwd("/home/s/Dropbox/KU/BSc Stas/R/ML_final_results/rolling_window_3")
ggsave(filename = "DJI_returns_backtest_rolling_3_ML.pdf", plot = DJI_plot, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
ggsave(filename = "GSPC_returns_backtest_rolling_3_ML.pdf", plot = GSPC_plot, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
ggsave(filename = "IXIC_returns_backtest_rolling_3_ML.pdf", plot = IXIC_plot, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
ggsave(filename = "N225_returns_backtest_rolling_3_ML.pdf", plot = N225_plot, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
ggsave(filename = "Joint_returns_rolling_3_ML.pdf", plot = plot_backtest, device = "pdf", dpi =500, width = 8, height = 5, units = "in")

dji_rmse_returns_roll_3 <- Metrics::rmse(DJI_returns_df$true_vals,DJI_returns_df$preds)
gspc_rmse_returns_roll_3   <- Metrics::rmse(GSPC_returns_df$true_vals,GSPC_returns_df$preds)
ixic_rmse_returns_roll_3  <- Metrics::rmse(IXIC_returns_df$true_vals,IXIC_returns_df$preds)
n225_rmse_returns_roll_3   <- Metrics::rmse(N225_returns_df$true_vals,N225_returns_df$preds)



rolling_3_returns_output_table<- data.frame(Series = c("Dow Jones","S&P 500","Nasdaq Composite","Nikkei 225"), 
                                            RMSE= c(dji_rmse_returns_roll_3, 
                                                    gspc_rmse_returns_roll_3,
                                                    ixic_rmse_returns_roll_3,
                                                    n225_rmse_returns_roll_3))
write.table(rolling_3_returns_output_table,"rolling_3_returns_output_table.txt",sep="\t",row.names=FALSE)
xtable(rolling_3_returns_output_table, digits = 8)

# Volatility ROLLING 3


DJI_volatility_df <- DJI_rolling_3_volatility_result[[1]] %>% mutate(Date= DJI_volatility_dates) 
GSPC_volatility_df <- GSPC_rolling_3_volatility_result[[1]]%>% mutate(Date= GSPC_volatility_dates) 


DJI_plot <- ggplot2::ggplot() + 
  geom_line(data = DJI_volatility_df, aes(x = Date, y = true_vals, color = "red",group = 1),size =0.5) +
  geom_line(data = DJI_volatility_df, aes(x = Date, y = preds, color = "darkblue",group = 1),size =0.5) +
  xlab('Time') +
  ylab('Realised Volatility')+scale_color_discrete(name = "Series", labels = c("Predicted","Actual"))+
  ggtitle("Dow Jones",subtitle = "Estimation on a rolling window of 3 years")+theme_bw()

GSPC_plot <- ggplot2::ggplot() + 
  geom_line(data = GSPC_volatility_df, aes(x = Date, y = true_vals, color = "red",group = 1),size =0.5) +
  geom_line(data = GSPC_volatility_df, aes(x = Date, y = preds, color = "darkblue",group = 1),size =0.5) +
  xlab('Time') +
  ylab('Realised Volatility')+scale_color_discrete(name = "Series", labels = c("Predicted","Actual"))+
  ggtitle("S&P500",subtitle = "Estimation on a rolling window of 3 years")+theme_bw()

plot_backtest <- grid.arrange(DJI_plot,
                              GSPC_plot, ncol=2)
setwd("/home/s/Dropbox/KU/BSc Stas/R/ML_final_results/rolling_window_3")
ggsave(filename = "DJI_volatility_rolling_3_ML.pdf", plot = DJI_plot, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
ggsave(filename = "GSPC_volatility_rolling_3_ML.pdf", plot = GSPC_plot, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
ggsave(filename = "Joint_volatility_rolling_3_ML.pdf", plot = plot_backtest, device = "pdf", dpi =500, width = 8, height = 5, units = "in")


dji_rmse_volatility_roll_3 <- Metrics::rmse(DJI_volatility_df$true_vals,DJI_volatility_df$preds)
gspc_rmse_volatility_roll_3   <- Metrics::rmse(GSPC_volatility_df$true_vals,GSPC_volatility_df$preds)

rolling_3_volatility_output_table<- data.frame(Series = c("Dow Jones","S&P 500"), 
                                            RMSE= c( dji_rmse_volatility_roll_3, 
                                                     gspc_rmse_volatility_roll_3
                                                     ))
write.table(rolling_3_volatility_output_table,"rolling_3_volatility_output_table.txt",sep="\t",row.names=FALSE)
xtable(rolling_3_volatility_output_table, digits = 8)

# Volatility ROLLING 1

DJI_volatility_df <- DJI_volatilit_2015_forecast_1yrs %>% mutate(Date= DJI_volatility_dates) 
GSPC_volatility_df <- GSPC_volatilit_2015_forecast_1yrs%>% mutate(Date= GSPC_volatility_dates) 



DJI_plot <- ggplot2::ggplot() + 
  geom_line(data = DJI_volatility_df, aes(x = Date, y = true_vals, color = "red",group = 1),size =0.5) +
  geom_line(data = DJI_volatility_df, aes(x = Date, y = preds, color = "darkblue",group = 1),size =0.5) +
  xlab('Time') +
  ylab('Realised Volatility')+scale_color_discrete(name = "Series", labels = c("Predicted","Actual"))+
  ggtitle("Dow Jones",subtitle = "Estimation on a rolling window of 1 year")+theme_bw()

GSPC_plot <- ggplot2::ggplot() + 
  geom_line(data = GSPC_volatility_df, aes(x = Date, y = true_vals, color = "red",group = 1),size =0.5) +
  geom_line(data = GSPC_volatility_df, aes(x = Date, y = preds, color = "darkblue",group = 1),size =0.5) +
  xlab('Time') +
  ylab('Realised Volatility')+scale_color_discrete(name = "Series", labels = c("Predicted","Actual"))+
  ggtitle("S&P500",subtitle = "Estimation on a rolling window of 1 year")+theme_bw()

plot_backtest <- grid.arrange(DJI_plot,
                              GSPC_plot, ncol=2)
setwd("/home/s/Dropbox/KU/BSc Stas/R/ML_final_results/rolling_window_1")
ggsave(filename = "DJI_volatility_rolling_1_ML.pdf", plot = DJI_plot, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
ggsave(filename = "GSPC_volatility_rolling_1_ML.pdf", plot = GSPC_plot, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
ggsave(filename = "Joint_volatility_rolling_1_ML.pdf", plot = plot_backtest, device = "pdf", dpi =500, width = 8, height = 5, units = "in")


dji_rmse_volatility_roll_1 <- Metrics::rmse(DJI_volatility_df$true_vals,DJI_volatility_df$preds)
gspc_rmse_volatility_roll_1   <- Metrics::rmse(GSPC_volatility_df$true_vals,GSPC_volatility_df$preds)

rolling_1_volatility_output_table<- data.frame(Series = c("Dow Jones","S&P 500"), 
                                               RMSE= c( dji_rmse_volatility_roll_1, 
                                                        gspc_rmse_volatility_roll_1
                                               ))
write.table(rolling_1_volatility_output_table,"rolling_1_volatility_output_table.txt",sep="\t",row.names=FALSE)
xtable(rolling_3_volatility_output_table, digits = 8)

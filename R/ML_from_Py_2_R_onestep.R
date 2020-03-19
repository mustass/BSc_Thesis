rm(list = ls())  
library(reticulate)
  library(ggplot2)
  library(gridExtra)
  library(xtable)
  use_condaenv(condaenv = "skorch")
  save_path <- "/home/s/Dropbox/KU/BSc Stas/R/ML_results/one_step_2015"
  py_run_string("sys.path.append('/home/s/Dropbox/KU/BSc Stas/Python/Try_again/')")
  
  source_python('/home/s/Dropbox/KU/BSc Stas/Python/Try_again/core/run_one_step_results_same_window.py')
  DJI_one_step_result <- results("DJI")
  GSPC_one_step_result <- results("GSPC")
  N225_one_step_result <- results("N225")
  IXIC_one_step_result <- results("IXIC")
  
  DJI_one_step_result_vol <-  results("DJI","volatility")
  GSPC_one_step_result_vol <- results("GSPC","volatility")
  
  # Correct dates:
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
  N225_csv[N225_csv == "null"] <- NA
  N225_csv <- N225_csv %>% na.omit()
  N225_csv$Datetime <- as.Date(N225_csv$Datetime)
  
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
  ####
  
  DJI_returns_df <- DJI_one_step_result[[1]] %>%  filter(dates>="2015-03-10") %>% mutate(dates = DJI_returns_dates) 
  GSPC_returns_df <- GSPC_one_step_result[[1]] %>%  filter(dates>="2015-03-06") %>% mutate(dates = GSPC_returns_dates) 
  IXIC_returns_df <- IXIC_one_step_result[[1]] %>%  filter(dates>="2015-03-11") %>% mutate(dates = IXIC_returns_dates) 
  N225_returns_df <- N225_one_step_result[[1]] %>%  filter(dates>="2015-02-17") %>% mutate(dates = N225_returns_dates) 
  
  GSPC_volatility_df <- GSPC_one_step_result_vol[[1]] %>% filter(dates>="2018-02-16") %>% mutate(dates = GSPC_volatility_dates)
  DJI_volatility_df <- DJI_one_step_result_vol[[1]] %>% filter(dates>="2018-03-02") %>% mutate(dates = DJI_volatility_dates)
  
  DJI_plot <- ggplot2::ggplot() + 
    geom_line(data = DJI_returns_df, aes(x = dates, y = true_vals, color = "red"),size =0.5) +
    geom_line(data = DJI_returns_df, aes(x = dates, y = preds, color = "darkblue"),size =0.5) +
    xlab('Time') +
    ylab('Log-returns')+scale_color_discrete(name = "Series", labels = c("Predicted","Actual"))+
    ggtitle("Dow Jones",subtitle = "Pre-estimated model")+theme_bw()
  
  GSPC_plot <- ggplot2::ggplot() + 
    geom_line(data = GSPC_returns_df, aes(x = dates, y = true_vals, color = "red"),size =0.5) +
    geom_line(data = GSPC_returns_df, aes(x = dates, y = preds, color = "darkblue"),size =0.5) +
    xlab('Time') +
    ylab('Log-returns')+scale_color_discrete(name = "Series", labels = c("Predicted","Actual"))+
    ggtitle("S&P500",subtitle = "Pre-estimated model")+theme_bw()
  
  IXIC_plot <- ggplot2::ggplot() + 
    geom_line(data = IXIC_returns_df, aes(x = dates, y = true_vals, color = "red"),size =0.5) +
    geom_line(data = IXIC_returns_df, aes(x = dates, y = preds, color = "darkblue"),size =0.5) +
    xlab('Time') +
    ylab('Log-returns')+scale_color_discrete(name = "Series", labels = c("Predicted","Actual"))+
    ggtitle("Nasdaq Composite",subtitle = "Pre-estimated model")+theme_bw()
  
  N225_plot <- ggplot2::ggplot() + 
    geom_line(data = N225_returns_df, aes(x = dates, y = true_vals, color = "red"),size =0.5) +
    geom_line(data = N225_returns_df, aes(x = dates, y = preds, color = "darkblue"),size =0.5) +
    xlab('Time') +
    ylab('Log-returns')+scale_color_discrete(name = "Series", labels = c("Predicted","Actual"))+
    ggtitle("Nikkei 225",subtitle = "Pre-estimated model")+theme_bw()
  
  plot_backtest <- grid.arrange(DJI_plot,
                                GSPC_plot,
                                IXIC_plot,
                                N225_plot , ncol=2)
  save_path <- "/home/s/Dropbox/KU/BSc Stas/R/ML_final_results/preEst"
  setwd(save_path)
  ggsave(filename =  "DJI_returns_ML_preest.pdf", plot = DJI_plot, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
  ggsave(filename = "GSPC_returns_ML_preest.pdf", plot = GSPC_plot, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
  ggsave(filename = "IXIC_returns_ML_preest.pdf", plot = IXIC_plot, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
  ggsave(filename = "N225_returns_ML_preest.pdf", plot = N225_plot, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
  ggsave(filename = "Joint_returns_ML_preest.pdf", plot = plot_backtest, device = "pdf", dpi =500, width = 8, height = 5, units = "in")
  
  dji_rmse_returns_preest  <-   Metrics::rmse(DJI_returns_df$true_vals,DJI_returns_df$preds)
  gspc_rmse_returns_preest   <- Metrics::rmse(GSPC_returns_df$true_vals,GSPC_returns_df$preds)
  ixic_rmse_returns_preest  <-  Metrics::rmse(IXIC_returns_df$true_vals,IXIC_returns_df$preds)
  n225_rmse_returns_preest  <-  Metrics::rmse(N225_returns_df$true_vals,N225_returns_df$preds)
  
  
  
  simple_backtest_output_table <- data.frame(Series = c("Dow Jones","S&P 500","Nasdaq Composite","Nikkei 225"), 
                                            RMSE= c(dji_rmse_returns_preest, 
                                                    gspc_rmse_returns_preest,
                                                    ixic_rmse_returns_preest,
                                                    n225_rmse_returns_preest))
  write.table(simple_backtest_output_table,"simple_backtest_output_table.txt",sep="\t",row.names=FALSE)
  xtable(simple_backtest_output_table, digits = 8)
  
  
  DJI_vol_plot <- ggplot2::ggplot() + 
    geom_line(data = DJI_volatility_df, aes(x = dates, y = true_vals, color = "red"),size =0.5) +
    geom_line(data = DJI_volatility_df, aes(x = dates, y = preds, color = "darkblue"),size =0.5) +
    xlab('Time') +
    ylab('Realised volatility')+scale_color_discrete(name = "Series", labels = c("Predicted","Actual"))+
    ggtitle("Dow Jones",subtitle = "Pre-estimated model")+theme_bw()
  
  GSPC_vol_plot <- ggplot2::ggplot() + 
    geom_line(data = GSPC_volatility_df, aes(x = dates, y = true_vals, color = "red"),size =0.5) +
    geom_line(data = GSPC_volatility_df, aes(x = dates, y = preds, color = "darkblue"),size =0.5) +
    xlab('Time') +
    ylab('Realised volatility')+scale_color_discrete(name = "Series", labels = c("Predicted","Actual"))+
    ggtitle("S&P500",subtitle = "Pre-estimated model")+theme_bw()
  
  plot_backtest_vol <- grid.arrange(DJI_vol_plot,
                                GSPC_vol_plot , ncol=2)
  save_path <- "/home/s/Dropbox/KU/BSc Stas/R/ML_final_results/preEst"
  setwd(save_path)
  ggsave(filename = "DJI_vol_preest_ML.pdf", plot = DJI_vol_plot, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
  ggsave(filename = "GSPC_vol_preest_ML.pdf", plot = GSPC_vol_plot, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")
  ggsave(filename = "Joint_vol_preest_ML_joint.pdf", plot = plot_backtest_vol, device = "pdf", dpi =500, width = 8, height = 5, units = "in")
  
   dji_rmse_volatility_preest  <-   Metrics::rmse(DJI_volatility_df$true_vals,DJI_volatility_df$preds)
  gspc_rmse_volatility_preest   <- Metrics::rmse(GSPC_volatility_df$true_vals,GSPC_volatility_df$preds)
  
  
  simple_backtest_output_table_vol <- data.frame(Series = c("Dow Jones","S&P 500"), 
                                             RMSE= c(dji_rmse_volatility_preest, 
                                                     gspc_rmse_volatility_preest )
                                                     )
  write.table(simple_backtest_output_table_vol,"simple_backtest_output_table_vol.txt",sep="\t",row.names=FALSE)
  xtable(simple_backtest_output_table_vol, digits = 8)
  
  
  

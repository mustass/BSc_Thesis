# Plot for choosing epochs
rm(list = ls())  
library(reticulate)
library(ggplot2)
library(gridExtra)
library(xtable)
use_condaenv(condaenv = "skorch")
save_path <- "/home/s/Dropbox/KU/BSc Stas/R/ML_results/one_step_2015"
py_run_string("sys.path.append('/home/s/Dropbox/KU/BSc Stas/Python/Try_again/')")

source_python('/home/s/Dropbox/KU/BSc Stas/Python/Try_again/core/run_one_step.py')
returns_epochs_results <- run_one_step("DJI", "returns")

df_epoch_returns <- data.frame(epochs = seq(1,200), train_loss = returns_epochs_results[[1]],
                               test_loss = returns_epochs_results[[2]])
returns_epochs <- ggplot(df_epoch_returns, aes(x = epochs))+
  geom_line(aes(y = train_loss,color ="red" ))+
  geom_line(aes(y = test_loss,color ="darkblue")) +
  geom_vline(xintercept = 11, color = "black",size = 0.5,linetype="dotted")+
  xlab("Number of epochs")+
  ylab("MSE")+theme_bw()+
  scale_color_discrete(name = "Subsets", labels = c("Validation","Training"))+
  ggtitle( "MSE as a function of epochs","DJI log-returns series")

ggsave(filename = "/home/s/Dropbox/KU/BSc Stas/Python/Try_again/results/volatility/one_step_epochs/DJI/returns_epochs.pdf", plot = returns_epochs, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")

vol_epochs_results <- run_one_step("DJI", "volatility")
  
df_epoch_vol <- data.frame(epochs = seq(1,200), train_loss = vol_epochs_results[[1]],
                               test_loss = vol_epochs_results[[2]])
vol_epochs <- ggplot(df_epoch_vol, aes(x = epochs))+
  geom_line(aes(y = train_loss,color ="red" ))+
  geom_line(aes(y = test_loss,color ="darkblue")) +
  geom_vline(xintercept = 10, color = "black", linetype="dotted", size = 0.5)+
  xlab("Number of epochs")+
  ylab("MSE")+theme_bw()+
  scale_color_discrete(name = "Subsets", labels = c("Validation","Training"))+
  ggtitle( "MSE as a function of epochs","DJI realised volatility series")+ylim(c(0,1E-06))
ggsave(filename = "/home/s/Dropbox/KU/BSc Stas/Python/Try_again/results/volatility/one_step_epochs/DJI/volatility_epochs.pdf", plot = vol_epochs, device ="pdf", dpi = 300, width =6 , height = 4, units = "in")

plot_epochs <- grid.arrange(returns_epochs,
                              vol_epochs,ncol=2)
ggsave(filename =  "/home/s/Dropbox/KU/BSc Stas/Python/Try_again/results/volatility/one_step_epochs/DJI/joint_epochs_plot.pdf", plot = plot_epochs, device = "pdf", dpi =500, width = 8, height = 5, units = "in")
  sqrt(df_epoch_vol$test_loss[10])
list <- list(df_epoch_returns,df_epoch_vol)
save(list, file ="/home/s/Dropbox/KU/BSc Stas/Python/Try_again/results/volatility/one_step_epochs/DJI/epochs_results.Rda")

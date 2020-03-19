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
save_path <- "/home/s/Dropbox/KU/BSc Stas/R/desc_stats"
dt <- read.csv("/home/s/Dropbox/KU/BSc Stas/R/data/data_volatility/oxfordmanrealizedvolatilityindices.csv",stringsAsFactors = F)
colnames(dt)[1] <- "Datetime"
dt$Datetime <- as.Date(dt$Datetime)

unique(dt$Symbol)
library(dplyr)
DJI_volatility <- dt %>% filter(Symbol == ".DJI") %>%  head(-1)
GSPC_volatility <- dt %>% filter(Symbol == ".SPX") %>% head(-1)


GSPC_plot <- ggplot(GSPC_volatility[GSPC_volatility$Datetime >"2018-02-15",]) +  aes(Datetime, rv5)+geom_line(color ='#009999' )+
  ggtitle("S&P 500") +
  ylab("Realised volatility") +
  xlab("Time")+theme_bw()
DJI_plot <-ggplot(DJI_volatility[DJI_volatility$Datetime >"2018-02-15",]) +  aes(Datetime, rv5)+geom_line(color ='#009999' )+
  ggtitle("Dow Jones Industrial Average") +
  ylab("Realised volatility ") +
  xlab("Time")+theme_bw()
plot <- grid.arrange(GSPC_plot, DJI_plot  , ncol=2)
ggsave(filename = "desc_stats/volatility_DJI_GSPC.pdf", plot = plot, device ="pdf", dpi = 300,width = 8, height = 5, units = "in")

GSPC_desc <- describe(GSPC_volatility$rv5)
N225_desc <- describe(N225_volatility$rv5)
DJI_desc <- describe(DJI_volatility$rv5)
IXIC_desc <- describe(IXIC_volatility$rv5)

desc_table <- GSPC_desc %>%  rbind(DJI_desc) %>% mutate(Series = c("S&P 500", "Dow Jones")) %>% select(Series,n,min,median,mean,max,range,skew,kurtosis,sd)
rownames(desc_table) <- NULL
xtable(desc_table, caption = "Summary of data", digits = 2)


hist_GSPC <- ggplot(GSPC_volatility, aes(x=rv5)) + 
  geom_histogram(colour="black", fill="white",binwidth = 0.000005) +theme_bw()+  ggtitle("S&P 500") +
  ylab("Frequency") +xlab("")+xlim(c(0,0.0005))

hist_DJI <- ggplot(DJI_volatility, aes(x=rv5)) + 
  geom_histogram(colour="black", fill="white",binwidth = 0.000005) +theme_bw()+  ggtitle("Dow Jones Industrial Average") +
  ylab("Frequency") +xlab("")+xlim(c(0,0.0005))


hists <- grid.arrange(hist_DJI,hist_GSPC , ncol=2)
ggsave(filename = "desc_stats/histogram_volatility_2.pdf", plot = hists, device ="pdf", dpi = 300, width = 8, height = 5, units = "in")

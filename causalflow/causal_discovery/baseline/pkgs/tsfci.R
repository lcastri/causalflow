#!/usr/bin/env Rscript

args = commandArgs(trailingOnly=TRUE)
data <- read.csv(file = as.character(args[1]))
data$time_index <- NULL
alpha <- as.double(args[2])
n_lags <- as.numeric(args[3])
path <- as.character(args[4])

print(path)
print("#######################")
setwd(path)

start_up <- function() {
  setwd("./pkgs/R_packages/ts-FCI_RCode_TETRADjar/")
  source('dconnected.R')
  source('genData.R')
  source('main_tetrad_fci.R')
  source('plot_timeseries.R')
  source('Plotting_Commands_Barplots.R')
  source('plot_ts_pag.R')
  source('realData_tsfci.R')
  source('scores.R')
  source('Simulation_Commands.R')
  source('Simulations_data_cont.R')
  source('Simulations_data_disc.R')
  source('Simulations_graph.R')
  source('Tetrad_R_interact.R')
  source('ts_functions.R')
}
start_up()


result <- realData_tsfci(data=data, sig=alpha, nrep=n_lags, inclIE=FALSE, alg="tscfci", datatype="continuous", makeplot=FALSE)
temporal_names = c()
for (i in 1:n_lags){
  for (name in colnames(data)){
    temporal_names <- c(temporal_names, paste(name,i-1, sep = "_"))
  }
}
colnames(result) = temporal_names
setwd(path)
# %write.csv(result, file="./results/result.csv", row.names = FALSE)
write.table(result, "./results/result.csv", col.names = temporal_names, row.names = temporal_names, sep = ",")





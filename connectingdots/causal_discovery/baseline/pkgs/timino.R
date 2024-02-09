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
    if (!requireNamespace("gam", quietly = TRUE)) {
      install.packages('gam', dependencies = TRUE, repos='http://cran.rstudio.com/')
    }
    if (!requireNamespace("kernlab", quietly = TRUE)) {
      install.packages('kernlab', dependencies = TRUE, repos='http://cran.rstudio.com/')
    }
    if (!requireNamespace("gptk", quietly = TRUE)) {
      # Download package tarball from CRAN archive
      url <- "https://cran.r-project.org/src/contrib/Archive/gptk/gptk_1.08.tar.gz"
      pkgFile <- "gptk_1.08.tar.gz"
      download.file(url = url, destfile = pkgFile)

      install.packages(c("Matrix", "fields", "spam"))

      # Install package
      install.packages(pkgs=pkgFile, type="source", repos=NULL)

      # Delete package tarball
      unlink(pkgFile)
    }

    library(gam)
    library(kernlab)
    library(gptk)
    source("./R_packages/codeTimino/timino_causality.R")
    source("./R_packages/codeTimino/util/hammingDistance.R")
    source("./R_packages/codeTimino/util/indtestAll.R")
    source("./R_packages/codeTimino/util/indtestHsic.R")
    source("./R_packages/codeTimino/util/indtestPcor.R")
    source("./R_packages/codeTimino/util/TSindtest.R")
    source("./R_packages/codeTimino/util/fitting_ts.R")
}
start_up()

result <- timino_dag(data, alpha = alpha, max_lag = n_lags, model = traints_linear, indtest = indtestts_crosscov, output = TRUE)

result[is.na(result)] <- 3

for (j1 in 1:nrow(result)){
    for (j2 in 1:nrow(result)){
      if (result[j1,j2] == 1){
        result[j1,j2] <- 2

      }
    }
}

for (j1 in 1:nrow(result)){
    for (j2 in 1:nrow(result)){
      if (result[j1,j2] == 2){
        if (result[j2,j1] == 0){
            result[j2,j1] <- 1
        }
      }
      if (j1 == j2){
          result[j1,j2] <- 1
      }
    }
}

for (j1 in 1:nrow(result)){
    for (j2 in 1:nrow(result)){
      if (result[j1,j2] == 3){
        result[j1,j2] <- 2
        result[j2,j1] <- 2
      }
    }
}


#write.csv(result, file="./results/result.csv", row.names = FALSE, col.names = FALSE)
write.table(result, "./results/result.csv", col.names = colnames(data), row.names = colnames(data), sep = ",")


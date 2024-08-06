library(emmeans)
library(ez)
library(effectsize)
library(readr)

resfolder <- "git/causalflow/results/AIS_major"
resname <- "AIS_major_S5"

graph_fpr <- read_csv(file.path(resfolder, resname, "graph_fpr_boot.csv"))
uncertainty <- read_csv(file.path(resfolder, resname, "uncertainty_boot.csv"))
pag_size <- read_csv(file.path(resfolder, resname, "pag_size_boot.csv"))
graph_shd <- read_csv(file.path(resfolder, resname, "graph_shd_boot.csv"))
graph_f1_score <- read_csv(file.path(resfolder, resname, "graph_f1_score_boot.csv"))
time <- read_csv(file.path(resfolder, resname, "time_boot.csv"))

# FPR
print("=======================================================================")
print("FPR analysis")
if (substr(resname, nchar(resname) - 1, nchar(resname)) %in% c("S3", "S5")) {
    model_fpr<-lm(graph_fpr~as.factor(algo)+nint,data=graph_fpr)
} else {
    model_fpr<-lm(graph_fpr~as.factor(algo)+nvars,data=graph_fpr)
}
print(summary(model_fpr))
print(emmeans(model_fpr,~algo))
print(eta_squared(model_fpr,ci=0.95,alternative = "greater"))

# Uncertainty
print("=======================================================================")
print("Uncertainty analysis")
if (substr(resname, nchar(resname) - 1, nchar(resname)) %in% c("S3", "S5")) {
    model_nSpurious<-lm(uncertainty~as.factor(algo)+nint,data=uncertainty)
} else {
    model_nSpurious<-lm(uncertainty~as.factor(algo)+nvars,data=uncertainty)
}
print(summary(model_nSpurious))
print(emmeans(model_nSpurious,~algo))
print(eta_squared(model_nSpurious,ci=0.95,alternative = "greater"))

# PAG Size
print("=======================================================================")
print("PAG Size analysis")
if (substr(resname, nchar(resname) - 1, nchar(resname)) %in% c("S3", "S5")) {
    model_equDAGs<-lm(pag_size~as.factor(algo)+nint,data=pag_size)
} else {
    model_equDAGs<-lm(pag_size~as.factor(algo)+nvars,data=pag_size)
}
print(summary(model_equDAGs))
print(emmeans(model_equDAGs,~algo))
print(eta_squared(model_equDAGs,ci=0.95,alternative = "greater"))

# SHD
print("=======================================================================")
print("SHD analysis")
if (substr(resname, nchar(resname) - 1, nchar(resname)) %in% c("S3", "S5")) {
    model_shd<-lm(graph_shd~as.factor(algo)+nint,data=graph_shd)
} else {
    model_shd<-lm(graph_shd~as.factor(algo)+nvars,data=graph_shd)
}
print(summary(model_shd))
print(emmeans(model_shd,~algo))
print(eta_squared(model_shd,ci=0.95,alternative = "greater"))

# F1-Score
print("=======================================================================")
print("F1-Score analysis")
if (substr(resname, nchar(resname) - 1, nchar(resname)) %in% c("S3", "S5")) {
    model_f1<-lm(graph_f1_score~as.factor(algo)+nint,data=graph_f1_score)
} else {
    model_f1<-lm(graph_f1_score~as.factor(algo)+nvars,data=graph_f1_score)
}
print(summary(model_f1))
print(emmeans(model_f1,~algo))
print(eta_squared(model_f1,ci=0.95,alternative = "greater"))

# Time
print("=======================================================================")
print("Time analysis")
if (substr(resname, nchar(resname) - 1, nchar(resname)) %in% c("S3", "S5")) {
    model_time<-lm(time~as.factor(algo)+nint,data=time)
} else {
    model_time<-lm(time~as.factor(algo)+nvars,data=time)
}
print(summary(model_time))
print(emmeans(model_time,~algo))
print(eta_squared(model_time,ci=0.95,alternative = "greater"))

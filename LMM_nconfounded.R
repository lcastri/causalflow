library(emmeans)
library(ez)
library(effectsize)

# Load the bootstrapped CSV files for each metric
fpr <- read.csv("~/git/connectingdots/results/rebuttal/nconfounded_nonlin_1250_250/fpr_bootstrapped.csv")
N_SpuriousLinks <- read.csv("~/git/connectingdots/results/rebuttal/nconfounded_nonlin_1250_250/N_SpuriousLinks_bootstrapped.csv")
N_EquiDAG <- read.csv("~/git/connectingdots/results/rebuttal/nconfounded_nonlin_1250_250/N_EquiDAG_bootstrapped.csv")
shd <- read.csv("~/git/connectingdots/results/rebuttal/nconfounded_nonlin_1250_250/shd_bootstrapped.csv")
f1_score <- read.csv("~/git/connectingdots/results/rebuttal/nconfounded_nonlin_1250_250/f1_score_bootstrapped.csv")
time <- read.csv("~/git/connectingdots/results/rebuttal/nconfounded_nonlin_1250_250/time_bootstrapped.csv")


# 0 - new
# 1 - FPCMCI
# 2 - PCMCI

# FPR
print("=======================================================================")
print("FPR analysis")
model_fpr<-lm(fpr~as.factor(algo)+nconfounded,data=fpr)
print(summary(model_fpr))
print(emmeans(model_fpr,~algo))
print(eta_squared(model_fpr,ci=0.95,alternative = "greater"))

# N_SpuriousLinks
print("=======================================================================")
print("N_SpuriousLinks analysis")
model_nSpurious<-lm(N_SpuriousLinks~as.factor(algo)+nconfounded,data=N_SpuriousLinks)
print(summary(model_nSpurious))
print(emmeans(model_nSpurious,~algo))
print(eta_squared(model_nSpurious,ci=0.95,alternative = "greater"))

# N_EquiDAG
print("=======================================================================")
print("Equ. DAGs analysis")
model_equDAGs<-lm(N_EquiDAG~as.factor(algo)+nconfounded,data=N_EquiDAG)
print(summary(model_equDAGs))
print(emmeans(model_equDAGs,~algo))
print(eta_squared(model_equDAGs,ci=0.95,alternative = "greater"))

# SHD
print("=======================================================================")
print("SHD analysis")
model_shd<-lm(shd~as.factor(algo)+nconfounded,data=shd)
print(summary(model_shd))
print(emmeans(model_shd,~algo))
print(eta_squared(model_shd,ci=0.95,alternative = "greater"))

# F1-Score
print("=======================================================================")
print("F1-Score analysis")
model_f1<-lm(f1_score~as.factor(algo)+nconfounded,data=f1_score)
print(summary(model_f1))
print(emmeans(model_f1,~algo))
print(eta_squared(model_f1,ci=0.95,alternative = "greater"))

# Time
print("=======================================================================")
print("Time analysis")
model_time<-lm(time~as.factor(algo)+nconfounded,data=time)
print(summary(model_time))
print(emmeans(model_time,~algo))
print(eta_squared(model_time,ci=0.95,alternative = "greater"))
library(emmeans)
library(ez)
library(effectsize)

# 0 - new
# 1 - FPCMCI
# 2 - PCMCI

# FPR
print("=======================================================================")
print("FPR analysis")
model_fpr<-lm(fpr~as.factor(algo)+nconfounded,data=fpr)
# model_fpr<-lm(fpr~as.factor(algo)+nvars,data=fpr)
print(summary(model_fpr))
print(emmeans(model_fpr,~algo))
print(eta_squared(model_fpr,ci=0.95,alternative = "greater"))

# N_SpuriousLinks
print("=======================================================================")
print("N_SpuriousLinks analysis")
model_nSpurious<-lm(N_SpuriousLinks~as.factor(algo)+nconfounded,data=N_SpuriousLinks)
# model_nSpurious<-lm(N_SpuriousLinks~as.factor(algo)+nvars,data=N_SpuriousLinks)
print(summary(model_nSpurious))
print(emmeans(model_nSpurious,~algo))
print(eta_squared(model_nSpurious,ci=0.95,alternative = "greater"))

# N_EquiDAG
print("=======================================================================")
print("Equ. DAGs analysis")
model_equDAGs<-lm(N_EquiDAG~as.factor(algo)+nconfounded,data=N_EquiDAG)
# model_equDAGs<-lm(N_EquiDAG~as.factor(algo)+nvars,data=N_EquiDAG)
print(summary(model_equDAGs))
print(emmeans(model_equDAGs,~algo))
print(eta_squared(model_equDAGs,ci=0.95,alternative = "greater"))

# SHD
print("=======================================================================")
print("SHD analysis")
model_shd<-lm(shd~as.factor(algo)+nconfounded,data=shd)
# model_shd<-lm(shd~as.factor(algo)+nvars,data=shd)
print(summary(model_shd))
print(emmeans(model_shd,~algo))
print(eta_squared(model_shd,ci=0.95,alternative = "greater"))

# F1-Score
print("=======================================================================")
print("F1-Score analysis")
model_f1<-lm(f1_score~as.factor(algo)+nconfounded,data=f1_score)
# model_f1<-lm(f1_score~as.factor(algo)+nvars,data=f1_score)
print(summary(model_f1))
print(emmeans(model_f1,~algo))
print(eta_squared(model_f1,ci=0.95,alternative = "greater"))

# Time
print("=======================================================================")
print("Time analysis")
model_time<-lm(time~as.factor(algo)+nconfounded,data=time)
# model_time<-lm(time~as.factor(algo)+nvars,data=time)
print(summary(model_time))
print(emmeans(model_time,~algo))
print(eta_squared(model_time,ci=0.95,alternative = "greater"))
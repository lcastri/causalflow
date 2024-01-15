library(emmeans)
library(ez)
library(effectsize)

# 0 - new
# 1 - FPCMCI
# 2 - PCMCI

# F1-SCORE
# print("=======================================================================")
# print("F1-Score analysis")
# model_f1<-lm(f1_score~as.factor(algo)+nvars,data=f1_score)
# summary(model_f1)
# emmeans(model_f1,~algo)
# eta_squared(model_f1,ci=0.95,alternative = "greater")

# SHD
print("=======================================================================")
print("SHD analysis")
model_shd<-lm(shd~as.factor(algo)+nvars,data=shd)
summary(model_shd)
emmeans(model_shd,~algo)
eta_squared(model_shd,ci=0.95,alternative = "greater")


# FPR
print("=======================================================================")
print("FPR analysis")
model_fpr<-lm(fpr~as.factor(algo)+nvars,data=fpr)
summary(model_fpr)
emmeans(model_fpr,~algo)
eta_squared(model_fpr,ci=0.95,alternative = "greater")

# nSpurious
print("=======================================================================")
print("nSpurious analysis")
model_nSpurious<-lm(nSpurious~as.factor(algo)+nvars,data=nSpurious)
summary(model_nSpurious)
emmeans(model_nSpurious,~algo)
eta_squared(model_nSpurious,ci=0.95,alternative = "greater")

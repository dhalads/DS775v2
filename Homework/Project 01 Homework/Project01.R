airfares = read.csv("data/Airfares.csv")
# data = select(airfares, "COUPON", "NEW", "VACATION", "SW", "HI", "S_INCOME", "E_INCOME", "S_POP", "E_POP", "SLOT", "GATE", "DISTANCE", "PAX", "FARE")
data = select(airfares, "COUPON", "VACATION", "SW", "HI", "S_INCOME", "E_INCOME", "S_POP", "E_POP", "SLOT", "GATE", "DISTANCE", "FARE")

summary(data)
data[,'VACATION']= as.factor(data[,'VACATION'])
data[,'SW']= as.factor(data[,'SW'])
data[,'SLOT']= as.factor(data[,'SLOT'])
data[,'GATE']= as.factor(data[,'GATE'])
table(data['VACATION'])
table(data['SW'])
summary(data)



library(leaps)
regfit = regsubsets(FARE ~ .   ,data = data, nvmax = 41) 

regfit

plot(regfit)
summary(regfit)

#Q4
plot(regfit, scale="adjr2")
regfit.summary = summary(regfit)
#Q5
which.max(regfit.summary$adjr2)
coef(regfit, 11)
#Q6
which.min(regfit.summary$bic)

result = lm(FARE~., data = data)
result
summary(result)

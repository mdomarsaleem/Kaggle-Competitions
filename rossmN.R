rm(list = ls(all=TRUE))
setwd("~/ROSSMAN")
library(sqldf)
library(xgboost)
library(readr)
train <- read.csv("train.csv",stringsAsFactors = F,header = T)
test <- read.csv("test.csv",stringsAsFactors = F,header = T)
store <- read.csv("store.csv",stringsAsFactors = F,header = T)
train$Date <- as.Date(train$Date)
test$Date <- as.Date(test$Date)
train$day <- as.numeric(as.factor(weekdays(train$Date)))
test$day <-  as.numeric(as.factor(weekdays(test$Date)))

# 
# 
# train$StateHoliday <-  as.factor(train$StateHoliday)
# test$StateHoliday <-  factor(test$StateHoliday,level=levels(train$StateHoliday))
# 
# train$month <- as.integer(format(train$Date, "%m"))
# train$year <- as.integer(format(train$Date, "%y"))
# train$Store <- as.factor(train$Store)
# 
# test$month <- as.integer(format(test$Date, "%m"))
# test$year <- as.integer(format(test$Date, "%y"))
# test$Store <- as.factor(test$Store)
# 

# a = sqldf('select store.Store,(train.Customers) from store left join train where store.Store = train.Store '  )
# b = sqldf('select Store,sum(Customers) from a group by Store')
# store= merge(store,b,by="Store",all.x = T)
# colnames(store)[colnames(store) == 'sum(Customers)'] <- 'Customer'
# store$StoreType = as.numeric(as.factor(store$StoreType))
# store$Assortment = as.numeric(as.factor(store$Assortment))
# store$PromoInterval = as.numeric(as.factor(store$PromoInterval))
# 



train <- merge(train,store)
test <- merge(test,store)

# There are some NAs in the integer columns so conversion to zero
train[is.na(train)]   <- 0
test[is.na(test)]   <- 0

cat("train data column names and details\n")
names(train)
str(train)
summary(train)
cat("test data column names and details\n")
names(test)
str(test)
summary(test)

# looking at only stores that were open in the train set
# may change this later
train <- train[ which(train$Open=='1'),]
train <- train[ which(train$Sales!='0'),]
# seperating out the elements of the date column for the train set
train$month <- as.integer(format(train$Date, "%m"))
train$year <- as.integer(format(train$Date, "%y"))
train$dat <- as.integer(format(train$Date, "%d"))
train$week <- trunc(train$dat/7)

# removing the date column (since elements are extracted) and also StateHoliday which has a lot of NAs (may add it back in later)
train <- train[,-c(3,8)]

# seperating out the elements of the date column for the test set
test$month <- as.integer(format(test$Date, "%m"))
test$year <- as.integer(format(test$Date, "%y"))
test$dat <- as.integer(format(test$Date, "%d"))
test$week <- trunc(test$dat/7)


# removing the date column (since elements are extracted) and also StateHoliday which has a lot of NAs (may add it back in later)
test <- test[,-c(4,7)]

feature.names <- names(train)[c(1,2,5:20)]
cat("Feature Names\n")
feature.names

cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

cat("train data column names after slight feature engineering\n")
names(train)
cat("test data column names after slight feature engineering\n")
names(test)
tra<-train[,feature.names]
RMPSE<- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  elab<-exp(as.numeric(labels))-1
  epreds<-exp(as.numeric(preds))-1
  err <- sqrt(mean((epreds/elab-1)^2))
  return(list(metric = "RMPSE", value = err))
}
nrow(train)
h<-sample(nrow(train),10000)

dval<-xgb.DMatrix(data=data.matrix(tra[h,]),label=log(train$Sales+1)[h])
dtrain<-xgb.DMatrix(data=data.matrix(tra[-h,]),label=log(train$Sales+1)[-h])
watchlist<-list(val=dval,train=dtrain)
param <- list(  objective           = "reg:linear", 
                booster = "gbtree",
                eta                 = 0.025, # 0.06, #0.01,
                max_depth           = 10, #changed from default of 8
                subsample           = 0.7, # 0.7
                colsample_bytree    = 0.7 # 0.7
                
                # alpha = 0.0001, 
                # lambda = 1
)

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 200, #300, #280, #125, #250, # changed from 300
                    verbose             = 0,
                    early.stop.round    = 100,
                    watchlist           = watchlist,
                    maximize            = FALSE,
                    feval=RMPSE
)
pred1 <- exp(predict(clf, data.matrix(test[,feature.names]))) -1
submission <- data.frame(Id=test$Id, Sales=pred1)
cat("saving the submission file\n")
write_csv(submission, "xgb.csv")



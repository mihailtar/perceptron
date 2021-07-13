library(data.table) # for quick learning data sets
library(dplyr) # for convinient working with derias of functions
library(ggplot2) # for plots
library(neuralnet) # for Neural Nets
library(DataExplorer) # for exploring data set
library(MLmetrics) # for machin elearning metrics
library(caret) # for preprocessing data sets

df<-fread(file = 'variant-7.csv',data.table = F)
head(df)
dim(df)
df %>% plot_intro()
summary(df)
df %>% plot_missing()
df<-data.frame(na.omit(df))
set.seed(0)
train_ind<-sample(x = 1:nrow(df),size = 0.7*nrow(df),replace = F)
# train and test data sets
train_data<-df[train_ind,]
test_data<-df[-train_ind,]
# create scaling model
preProcValues <- preProcess(train_data, method = c("center", "scale"))
# scale training predictors and response
train_data_trans <- predict(preProcValues, train_data)
# scale test predictors and response
test_data_trans <- predict(preProcValues, test_data)
set.seed(0)
m<-neuralnet(formula = "V1~.",
             data = train_data_trans, # training data
             hidden = c(5,5,3), # size hidden layers
             threshold = 0.001, # threshold
             stepmax = 100000, # max iterations
             err.fct = 'sse', # the error mesure
             linear.output = T, # linear output
             act.fct = 'logistic' # activation function
)
plot(x = m, show.weights = T, # print weights values?
     fontsize = 10, rep = "best")
# train approximation values
pr_train<-(predict(object = m,
                   newdata = train_data_trans) * preProcValues$std[1])+preProcValues$mean[1]
# test approximation values
pr_test<-(predict(object = m,
                  newdata = test_data_trans) * preProcValues$std[1])+preProcValues$mean[1]
# for training data set
r2_train<-MLmetrics::R2_Score(y_pred = pr_train,y_true = train_data$V1)
mae_train<-MLmetrics::MAE(y_pred = pr_train,y_true = train_data$V1)
mape_train<-100*MLmetrics::MAPE(y_pred = pr_train,y_true = train_data$V1)
# for testing data set
r2_test<-MLmetrics::R2_Score(y_pred = pr_test,y_true = test_data$V1)
mae_test<-MLmetrics::MAE(y_pred = pr_test,y_true = test_data$V1)
mape_test<-100*MLmetrics::MAPE(y_pred = pr_test,y_true = test_data$V1)
all_res<-data.frame(variant = c("train","test"),
                    r2 = c(r2_train,r2_test),
                    mae = c(mae_train,mae_test),
                    mape = c(mape_train,mape_test))
all_res
res<-data.frame(V1=test_data$V1,V1_pred = pr_test,Obs=1:length(pr_test))
head(res)
ggplot(data=res)+ 
  geom_line(aes(x=Obs, y=V1,colour = 'V1'))+
  geom_line(aes(x=Obs, y=V1_pred,colour = 'V1_pred'))+
  ggtitle('Истинные и прогнозные значения переменной V1')+
  labs(colour = "Легенда:")+
  theme_minimal()

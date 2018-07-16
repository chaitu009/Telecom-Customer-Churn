rm(list = ls())

#Load Libraries
x = c('ggplot2', 'corrgram', 'DMwR', 'caret', 'randomForest', 'unbalanced', 'C50', 'dummies', 
      'e1071', 'Information','MASS','rpart', 'gbm', 'ROSE') 

#install.packages(x)
lapply(x, require, character.only = TRUE)
rm(x)
library(ggplot2)
#Importing the dataset

setwd('C:/Users/chait/Desktop/Project1')
df = read.csv('Train_Data.csv')

#Data preprocessing
sum(is.na(df))
head(df)
str(df)
df$state = as.factor(as.character(df$state))
df$area.code = as.factor(as.character(df$area.code))
str(df$state)
str(df$area.code)
df = subset(df,select =-c(phone.number))

#Exploratory data analysis
library(reshape)
table(df$Churn)
pie(table(df$Churn),main = 'Pie chart for Customer Churn',radius = 1)

#Exploratory data analysis
imppred <- randomForest(Churn ~ ., data = df,
                        ntree = 100, keep.forest = FALSE, importance = TRUE)
importance(imppred, type = 1)

#Seperating the numeric data
numeric_index = sapply(df,is.numeric)
numeric_data = df[,numeric_index]

cnames = colnames(numeric_data)

#Encoding categorical variables 
for(i in 1:ncol(df)){
  if(class(df[,i]) == 'factor'){
    df[,i] = factor(df[,i], labels =(1:length(levels(factor(df[,i])))))
  }
}

#Outlier analysis
for (i in 1:length(cnames)){
  assign(paste0("gn",i),ggplot(aes_string(y = cnames[i],x = "Churn"),data = subset(df))+
           stat_boxplot(geom = "errorbar", width = 0.5)+
           geom_boxplot(outlier.colour = 'red', fill = "green", outlier.shape = 18,
                        outlier.size = 1, notch = FALSE) +
           theme(legend.position = "bottom")+
           labs(y=cnames[i],x = "Churn")+
           ggtitle(paste(" ", cnames[i])))
}
#Plotting plots together
gridExtra::grid.arrange(gn1,gn2,gn3,ncol = 3)
gridExtra::grid.arrange(gn4,gn5,gn6,ncol = 3)
gridExtra::grid.arrange(gn7,gn8,gn9,ncol = 3)
gridExtra::grid.arrange(gn10,gn11,gn12,ncol = 3)
gridExtra::grid.arrange(gn13,gn14,gn15,ncol = 3)

#Imputing the outlier values with knnImputation method
for( i in cnames){
  print(i)
  val = df[,i][df[,i] %in% boxplot.stats(df[,i])$out]
  df[,i][df[,i] %in% val] = NA
}

df = knnImputation(df, k=80)

#Feature Selection | Correlation plot for numeric variables
library(corrplot)

m = cor(df[,numeric_index])
corrplot(m, method="square")

#Feature Selection 
factor_index = sapply(df,is.factor)
factor_data = df[,factor_index]

for(i in 1:4)
{
  print(names(factor_data[i]))
  print(chisq.test(table(factor_data$Churn,factor_data[,i])))
  
}

#Feature Selection or Dimensionality Reduction
df1 = subset(df, select = -c(total.day.charge,total.eve.charge,total.night.charge,total.intl.charge,area.code,voice.mail.plan))

numeric_index = sapply(df1,is.numeric)
numeric_data = df1[,numeric_index]

df_train=df1

#Feature Scaling | Normalizing the data
for( i in colnames(numeric_data))
{
  print(i)
  df_train[,i] = (df_train[,i]- min(df_train[,i]))/
    (max(df_train[,i] - min(df_train[,i])))   
}

############ Test Data processing############

#Importing the test data
df = read.csv('Test_data.csv')
df=df[sample(nrow(df), 500), ]
write.csv(df,'sampleinput_R.csv')
sum(is.na(df))
head(df)
str(df)

#Basic pre-processing
df$state = as.factor(as.character(df$state))
df$area.code = as.factor(as.character(df$area.code))
str(df$area.code)
datatypes = sapply(df,class)
for(i in 1:ncol(df)){
  if(class(df[,i]) == 'factor'){
    df[,i] = factor(df[,i], labels =(1:length(levels(factor(df[,i])))))
  }
}

#Feature Selection or Dimensionality Reduction in Test Data
df1 = subset(df, select = -c(total.day.charge,total.eve.charge,total.night.charge,total.intl.charge,phone.number,area.code,voice.mail.plan))

df_test=df1

#Feature scaling | Normalization
for( i in colnames(numeric_data))
{
  print(i)
  df_test[,i] = (df_test[,i]- min(df_test[,i]))/
    (max(df_test[,i] - min(df_test[,i])))   
}

#Building and training a decision tree model
#install.packages('C50')
library(C50)
C50_model = C5.0(Churn ~.,df_train, trails = 100, rules = TRUE)

#summary(C50_model)
#Predicting the output
C50_predictions = predict(C50_model,df_test[,-15],type = 'class')
write.csv(C50_predictions,'output_R.csv')

#Evaluate the performance of classification model
Confmatrix_C50 = table(df_test$Churn,C50_predictions)
confusionMatrix(Confmatrix_C50)

#############################################END########################  

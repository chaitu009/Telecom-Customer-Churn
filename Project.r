rm(list = ls())
#Load Libraries
x = c('ggplot2', 'corrgram', 'DMwR', 'caret', 'randomForest', 'unbalanced', 'c50', 'dummies', 
      'e1071', 'Information','MASS','rpart', 'gbm', 'ROSE') 
install.packages(x)
lapply(x, require, character.only = TRUE)
rm(x)
library(ggplot2)

setwd('C:/Users/e.chaitanya.vutukuru/Desktop/Project')

df = read.csv('Train_data.csv')
sum(is.na(df))
head(df)
str(df)

df$state = as.factor(as.character(df$state))
df$area.code = as.factor(as.character(df$area.code))
str(df$area.code)
datatypes = sapply(df,class)
for(i in 1:ncol(df)){
    if(class(df[,i]) == 'factor'){
    df[,i] = factor(df[,i], labels =(1:length(levels(factor(df[,i])))))
  }
}

numeric_index = sapply(df,is.numeric)
numeric_data = df[,numeric_index]

cnames = colnames(numeric_data)
for (i in 1:length(cnames)){
  assign(paste0("gn",i),ggplot(aes_string(y = cnames[i],x = "Churn"),data = subset(df))+
           stat_boxplot(geom = "errorbar", width = 0.5)+
           geom_boxplot(outlier.colour = 'red', fill = "grey", outlier.shape = 18,
                        outlier.size = 1, notch = FALSE) +
           theme(legend.position = "bottom")+
           labs(y=cnames[i],x = "Churn")+
           ggtitle(paste(" ", cnames[i])))
}
#Plotting plots together
#gridExtra::grid.arrange(gn1,gn5,gn2,ncol = 3)
#gridExtra::grid.arrange(gn6,gn7,ncol = 2)
#gridExtra::grid.arrange(gn8,gn9,gn11,gn15,ncol = 5)

###for (i in cnames){
  #print(i)
  #val = df[,i][df[,i] %in% boxplot.stats(df[,i])$out]
  #print(length(val))
  #df = df[which(!df[,i] %in% val),]
  #}
for( i in cnames){
  print(i)
  val = df[,i][df[,i] %in% boxplot.stats(df[,i])$out]
  df[,i][df[,i] %in% val] = NA
}

df = knnImputation(df, k=80)
library(corrgram)
#write.csv(df,"data.csv")
#corrgram(df[,numeric_index], order = T,
    #     upper.panel =panel.pie, text.panel = panel.txt, main = "Correlation Plot")
#library(GGally)
#ggcorr(df[,numeric_index])

#install.packages("corrplot")
library(corrplot)

m = cor(df[,numeric_index])
corrplot(m, method="square")

factor_index = sapply(df,is.factor)
factor_data = df[,factor_index]

for(i in 1:(ncol(factor_data)-1))
{
  print(names(factor_data[i]))
  print(chisq.test(table(factor_data$Churn,factor_data[,i])))
}

df1 = subset(df, select = -c(total.day.charge,total.eve.charge,total.night.charge,total.intl.charge,phone.number,area.code,voice.mail.plan))

qqnorm(df1$total.day.minutes)
hist(df1$total.intl.minutes)

numeric_index = sapply(df1,is.numeric)
numeric_data = df1[,numeric_index]

df_train=df1
for( i in colnames(numeric_data))
{
  print(i)
  df_train[,i] = (df_train[,i]- min(df_train[,i]))/
            (max(df_train[,i] - min(df_train[,i])))   
}

############ Test Data############
df = read.csv('Test_data.csv')
sum(is.na(df))
head(df)
str(df)

df$state = as.factor(as.character(df$state))
df$area.code = as.factor(as.character(df$area.code))
str(df$area.code)
datatypes = sapply(df,class)
for(i in 1:ncol(df)){
  if(class(df[,i]) == 'factor'){
    df[,i] = factor(df[,i], labels =(1:length(levels(factor(df[,i])))))
  }
}

df1 = subset(df, select = -c(total.day.charge,total.eve.charge,total.night.charge,total.intl.charge,phone.number,area.code,voice.mail.plan))
df_test=df1
for( i in colnames(numeric_data))
{
  print(i)
  df_test[,i] = (df_test[,i]- min(df_test[,i]))/
    (max(df_test[,i] - min(df_test[,i])))   
}



#install.packages('C50')
library(C50)
C50_model = C5.0(Churn ~.,df_train, trails = 100, rules = TRUE)

summary(C50_model)

C50_predictions = predict(C50_model,df_test[,-15],type = 'class')

#Evaluate the performance of classification model
Confmatrix_C50 = table(df_test$Churn,C50_predictions)
confusionMatrix(Confmatrix_C50)


#RF
RF_model = randomForest(Churn ~.,df_train, importance = TRUE, ntree=100)
RF_Predictions = predict(RF_model, df_test[,-15])

Confmatrix_RF = table(df_test$Churn,RF_Predictions)
confusionMatrix(Confmatrix_RF)


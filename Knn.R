library(tidyverse)
library(dplyr)
library(Hmisc)
library(class)
library(mice)
library(Amelia)
library(NoiseFiltersR)
require(caret)
library(imbalance)
library(DMwR)
library(FSelector)
library(FactoMineR)
library(factoextra)
set.seed(100)


#Lectura de train
datosTrain <- read.csv("train.csv", na.strings=c(" ","NA","?"))
#Lectura de test
datosTest <- read.csv("test.csv", na.strings=c(" ","NA","?"))
#Convertir a tibble
datosTrain <- as.tibble(datosTrain)
datosTest <- as.tibble(datosTest)


##### Descripcion de los datos ####
#Dimensiones de los datos
dim(datosTrain)
dim(datosTest)
#Variables
colnames(datosTrain)
colnames(datosTest)
Hmisc::describe(datosTrain[,1:7])
unique(datosTrain["C"])
Hmisc::describe(datosTest[,1:7])


##### Datos perdidos ####
#Gráfico de los datos perdidos
patron <- mice::md.pattern(x=datosTrain,plot = TRUE)
#Distribución de instancias completas e incompletas
mice::ncc(datosTrain)
mice::nic(datosTrain)

dim(na.omit(datosTrain))



##### Imputacion ####
#Imputación mediante Amelia
imputados <- Amelia::amelia(datosTrain, m=3, parallel="multicore",noms="C")
datosTrain <- imputados$imputations$imp1
#Comprobación de datos incompletos
mice::nic(datosTrain)
dim(datosTrain)


##### Eliminación de ruido en los datos mediante IPF####
datosTrainConRuido <- datosTrain

#Se estructuran los datos para que sean aceptados por el método
datosTrain <- as.data.frame(datosTrain)
datosTrain[,ncol(datosTrain)] <- as.factor(datosTrain[,ncol(datosTrain)])

#Se realiza IPF
out <- IPF(C~.,data=datosTrain)
length(out$remIdx)

#Se actualizan los datos eliminando aquellas instancias detectadas como anomalías
datosTrain <- as.tibble(datosTrain[setdiff(1:nrow(datosTrain),out$remIdx),])
datosTrain[,ncol(datosTrain)] <- as.numeric(unlist(datosTrain[,ncol(datosTrain)]))




##### Detección de anomalías
#Se van a detectar el 10% de las instancias como outliers
numero.de.outliers = as.integer(dim(datosTrainConRuido)[1]/10)

#Se contruye un modelo con kmeans
modelo.kmeans <- kmeans(datosTrainConRuido,2)
indices.clustering <- modelo.kmeans$cluster
centroides <- modelo.kmeans$centers

#Se calcula la distancia de las instancias a los centroides de los clusters para saber cuáles son más ruidosas  
distancias_a_centroides = function (datos,indices.asignacion.clustering,datos.centroides){
      sqrt(rowSums(   (datos - datos.centroides[indices.asignacion.clustering,])^2   ))
}
dist.centroides <- distancias_a_centroides(datosTrainConRuido, indices.clustering, centroides)

#Se eliminan las instancias que estén a mayor distancia
top.outliers <- order(dist.centroides, decreasing = TRUE)[1:numero.de.outliers]
datosTrainConRuido <- datosTrainConRuido[-top.outliers,]
dim(datosTrainConRuido)



##### Tratamiento del ruido en los datos de test
#Se detectan que instacias de test son ruidosas
ruido <- which(datosTest[,1] < -68900)
ruido

#Se intercambian por la mediana del resto de valores
medianas <- apply(datosTest[-ruido,],2,median)
cambio <- matrix(rep(medianas,length(ruido)),length(ruido),ncol(datosTest),byrow=TRUE)
datosTest[ruido,]<-cambio



##### Normalización de los datos ####
#Se estima el centrado y escalado en función de los datos de entrenamiento
datosPreprocesados <- caret::preProcess(datosTrain[,1:50],method=c("center","scale"))

#Se centran y escalan los datos de entrenamiento
datosTransformados <- predict(datosPreprocesados,datosTrain[,1:50])

#Se añade la variable predecir que no se puede modificar
datosTrain <- cbind(datosTransformados,datosTrain[,51])

#Se centran y escalan los datos de test en función del centrado y escalado que se había calculado
datosTest <- predict(datosPreprocesados,datosTest)
datosTrain <- as.tibble(datosTrain)
datosTest <- as.tibble(datosTest)


##### Correlación entre variables ####
#Construcción matriz de correlación
limite <- ncol(datosTrain)-1
corrMatrix <- cor(na.omit(datosTrain[,1:limite]))
corrplot::corrplot(corrMatrix,order="FPC",type="upper",tl.col="black",tl.srt=45)

#Ejemplos de filtrado de correlación entre variables
altamenteCorreladas <- caret::findCorrelation(corrMatrix, cutoff=0.9)
length(altamenteCorreladas)
altamenteCorreladas <- caret::findCorrelation(corrMatrix, cutoff=0.99)
length(altamenteCorreladas)
altamenteCorreladas <- caret::findCorrelation(corrMatrix, cutoff=0.9999)
length(altamenteCorreladas)

#Se eliminan las variables que esten muy correladas con otras
datosTrainFiltrados <- datosTrain[,-altamenteCorreladas]
datosTestFiltrados <- datosTest[,-altamenteCorreladas]
dim(datosTrainFiltrados)
dim(datosTestFiltrados)




##### Selección de características ####
# CFS
subset <- FSelector::cfs(C ~ .,datosTrain)
subset

# Consistency
subset <- consistency(C~.,datosTrain)
subset

# Relief
weights <- FSelector::relief(C ~.,datosTrain,neighbours.count = 30,sample.size = 120)
subset <- FSelector::cutoff.k(weights,2)
subset


# PCA
limite <- ncol(datosTrain)-1
res.pca <- PCA(datosTrain[,1:limite],graph = FALSE)

#Se seleccionan aquellas componentes que conformen el 98% del total, consiguiendo así las
#componentes más importantes
topComponentes <- unname(which(res.pca$eig[,3]>98)[1])

#Para estas componentes escogidas, se obtiene la contribución de las variables a ellas  
contrib <- fviz_contrib(res.pca,choice="var",axes=1:topComponentes)

#Gracias a que se devuelven las variables ordenadas en función de su contribución, se 
#escogen los dos primeros tercios, es decir, el 66% de las variables que en función de
#su contribución a las componentes son más importantes
topVars <- as.integer(ncol(datosTrain)/1.5)
subset <- contrib$data$name[order(contrib$data$contrib,decreasing = TRUE)][1:topVars]
subset <- unlist(lapply(subset,toString))
subset


#Se escogen solamente las variables selccionadas en el dataset
datosTrain <- datosTrain[,c(subset,"C")]
datosTest <- datosTest[,subset]
datosTrain[,"C"] <- datosTrain[,"C"] - 1
dim(datosTrain)
dim(datosTest)



##### Balanceo de clases ####

#Distribución de las instancias segun la variable de salida
ggplot(data=datosTrain) + geom_bar(mapping =aes(x=C,y=..prop..,group=1))
resumen <- dplyr::group_by(datosTrain,C) %>% dplyr::summarise(nc=n())
resumen

#Se utiliza SMOTE para balancear las clases
datosTrainBalanceados <- datosTrain
datosTrainBalanceados$C <- as.factor(datosTrainBalanceados$C)
datosTrainBalanceados <- SMOTE(C~., as.data.frame(datosTrainBalanceados), perc.over=100)

resumen <- dplyr::group_by(datosTrainBalanceados,C) %>% dplyr::summarise(nc=n())
print(resumen)


#Ejemplo de balanceo de clases con MWMOTE
datosTrainBalanceados <- imbalance::oversample(as.data.frame(datosTrain), ratio = 0.85, method = "MWMOTE", classAttr = "C")
resumen <- dplyr::group_by(datosTrainBalanceados,C) %>% dplyr::summarise(nc=n())
print(resumen)

#Ejemplo de balanceo de clases con ADASYN
datosTrainBalanceados <- imbalance::oversample(as.data.frame(datosTrain), ratio = 0.85, method = "ADASYN", classAttr = "C")
resumen <- dplyr::group_by(datosTrainBalanceados,C) %>% dplyr::summarise(nc=n())
print(resumen)




##### Validación cruzada ####

#Se divide los datos de entrenamiento en pseudoTrain y pseudoTest a relación de 85%-15%
misDatosTrain <- datosTrain
shuffled <- sample(dim(misDatosTrain)[1])
eightypct <- (dim(misDatosTrain)[1] * 85) %/% 100
datosTrainPseudo <- misDatosTrain[shuffled[1:eightypct], 1:(dim(misDatosTrain)[2]-1)]
datosTestPseudo <- misDatosTrain[shuffled[(eightypct+1):dim(misDatosTrain)[1]], 1:(dim(misDatosTrain)[2]-1)]
datosTrainPseudo_labels <- misDatosTrain[shuffled[1:eightypct], dim(misDatosTrain)[2]]
datosTrainPseudo_labels <-as.data.frame(datosTrainPseudo_labels)[,1]
datosTestPseudo_labels <- misDatosTrain[shuffled[(eightypct+1):dim(misDatosTrain)[1]], dim(misDatosTrain)[2]]
datosTestPseudo_labels <- as.data.frame(datosTestPseudo_labels)[,1]


#Función que a partir de una 'k' dada, calcula el acierto del modelo con los datos de entrenamiento
getKnn <- function(miK){
     
  #Se entrena el modelo con la 'k' indicada                                                                                 
  test_pred <- knn(train = datosTrainPseudo, test = datosTestPseudo,
                   cl = datosTrainPseudo_labels, k=miK)
  
  #Se estructura la predicción realizada
  test_pred <- as.vector(test_pred)
  test_pred[which(is.na(test_pred))] <- 2
  test_pred <- as.numeric(test_pred)
  
  #Se contruye la matriz de confusión y se devuelve el acierto obtenido por el modelo
  u <- union(test_pred, datosTestPseudo_labels)
  t <- table(factor(test_pred, u), factor(datosTestPseudo_labels, u))
  cm<-confusionMatrix(t)
  cm$overall["Accuracy"]
  
}

#Se obtienen los aciertos utilizando k=[1,3,5,7,...,297,299]
result1 <- lapply(1:150*2-1,getKnn)

#Se construye un histograma con los resultados obtenidos con diferentes 'k'
r1 <- unlist(result1)
df <- data.frame(k=1:150*2-1,accuracy=r1)
ggplot(df, aes(x=k,y=accuracy)) + geom_histogram(stat="identity",color="black",
                                                 fill="deepskyblue")+coord_cartesian(ylim=c(0.8,0.95)) + labs(title="Acierto de kNN con diferentes k") 

#Mejores valores de 'k' con los aciertos obtenidos
(1:150*2-1)[order(r1,decreasing=TRUE)[1:30]]


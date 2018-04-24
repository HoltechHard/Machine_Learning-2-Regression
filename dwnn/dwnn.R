# ------------------------------------------------------
#      DISTANCE WEIGHTED - NEAREST NEIGHBORS (DW-NN)        
# ------------------------------------------------------

#calculo pesos
weight <- function(xi, xj, sigma){
  euclidean = sqrt(sum((xi - xj)^2))
  gauss = exp(-euclidean^2/(2 * sigma^2))
  return(gauss)
}

#algoritmo dw-nn
dwnn <- function(dataset, query, sigma = 1){
  
  #definicao de variaveis
  classId = ncol(dataset)
  x = as.matrix(dataset[, 1:(classId-1)])
  y = dataset[, classId]
  
  #funcao de distancia
  w = apply(x, 1, function(row){
    weight(query, row, sigma)
  })
  
  #predicao do valor y
  y_pred = sum(y * w) / sum(w)
  
  #resposta
  ret = list()
  ret$predict = y_pred
  
  return(ret)
}

#toy: 200 amostras
dwnn.toy <- function(sigma, train.size = 0.7){
  dataset = cbind(seq(-10, 10, len = 200), seq(-10, 10, len = 200))
  ids = sample(1:nrow(data), size = floor(train.size * nrow(data)))
  
  #conjunto treinamento - conjunto teste
  train.set = dataset[ids, ]
  test.set = dataset[-ids, ]
  
  #fase de treinamento/teste
  classId = ncol(test.set)
  error = 0
  
  for(i in 1:nrow(test.set)){
    x.test_i = test.set[i, 1:(classId-1)]
    y.test_i = test.set[i, classId]
    
    res = dwnn(dataset = train.set, query = x.test_i, sigma = sigma)
    
    error = error + (res$predict - y.test_i)^2
  }
  
  error = error / nrow(test.set)
  cat('error: ', error, '... \n')
}

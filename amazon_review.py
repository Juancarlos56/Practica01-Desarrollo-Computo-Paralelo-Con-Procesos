#!/usr/bin/env python
# coding: utf-8

# ## Carga de Dataset

def cargaDataSet(path, nameColumns):
    df = pd.read_csv(path, encoding='utf-8', sep=',', low_memory=False, header=None, names=nameColumns)
    df = df.dropna()
    print(df.isnull().sum())
    df['reviewText'].replace(regex=True, inplace=True, to_replace=r"@'[^\w\s.!@$%^&*()\-\/]+'", value=r' ')
    df['review'].replace(regex=True, inplace=True, to_replace=r"@'[^\w\s.!@$%^&*()\-\/]+'", value=r' ')
    df = df.sort_values(["score"])
    df=df.reset_index(drop=True)
    print(df.shape)
    print(df.score.value_counts())
    return df

def reparacionDataSet(df, columna, identificador):
    is_score = df.loc[:, columna] == identificador
    df_grupo = df.loc[is_score]
    print("Tamaño SubDataSet: ", df_grupo.shape)
    return df_grupo

# ## Generación de una bolsa de palabras de manera Secuencial 

# ## Generación de una bolsa de palabras de manera Paralela con procesos 

def tokenizer(text):
    return text.split(" ")

def tokenizer_porter(text):
    return [porter.stem(word) for word in tokenizer(text)]

def countWords(text,diccionarioFrecuencia):
    
    for w in tokenizer(text): 
        if w not in stop:
            cont = 0
            for k in diccionarioFrecuencia.keys():
                if (k == w):
                    diccionarioFrecuencia[w] = diccionarioFrecuencia.get(w)+1
                    cont= cont+1
            if (cont == 0 ):
                diccionarioFrecuencia[w] = 1
    return diccionarioFrecuencia

def imprimirTOP10(diccionario): 
    cont=0
    for key in diccionario.keys():
        print("\t Palabra: ", key, "\t Valor: ", diccionario.get(key))
        cont= cont+1
        if(cont ==20 ):
            break;
        
def obtenerValoresPorGrupos(diccionarioFrecuenciaPalabras, datasetGrupo):  
    #for fila in range(0,len(datasetGrupo)):
    for fila in range(0,10):
        diccionarioFrecuenciaPalabras = countWords(datasetGrupo.iloc[fila]['reviewText'],diccionarioFrecuenciaPalabras)
    #Ordenar de menor a mayor
    diccionarioFrecuenciaPalabras = {k: v for k, v in sorted(diccionarioFrecuenciaPalabras.items(), key=lambda item: item[1], reverse=True)}
    imprimirTOP10(diccionarioFrecuenciaPalabras)
    #return diccionarioFrecuenciaPalabras


#Importación de librerías
#from do_something import do_it_now
import time
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
## Importacion de paquetes para procesamiento de Lenguaje natural
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

stop = stopwords.words('english')

if __name__ == "__main__":
    
    nltk.download('stopwords')
    porter = PorterStemmer()
    
    
    start_time = time.time()
    df = cargaDataSet('amazon_review_full_csv/train.csv', ['score', 'review', 'reviewText'])
    
    df_grupo1 = pd.DataFrame()
    df_grupo2 = pd.DataFrame()
    df_grupo3 = pd.DataFrame()
    df_grupo4 = pd.DataFrame()
    df_grupo5 = pd.DataFrame()
    
    dicc_grupo1 = {}
    dicc_grupo2 = {}
    dicc_grupo3 = {}
    dicc_grupo4 = {}
    dicc_grupo5 = {}
    
    datasets = [df_grupo1, df_grupo2, df_grupo3, df_grupo4, df_grupo5]
    diccionarios = [dicc_grupo1, dicc_grupo2, dicc_grupo3, dicc_grupo4, dicc_grupo5]
    
    procs = 5   
    jobs = []
    cont = 0
    
    for dataset in datasets:
        dataset = reparacionDataSet(df, 'score', cont+1)
        process = multiprocessing.Process                  (target=obtenerValoresPorGrupos, args=(diccionarios[cont],dataset))
        jobs.append(process)
        cont = cont+1
        
    for j in jobs:
        j.start()

    for j in jobs:
        j.join()

    print ("Count processing complete.")
    end_time = time.time()
    print("multiprocesses time=", end_time - start_time)


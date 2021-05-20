# Practica01-Desarrollo-Computo-Paralelo-Con-Procesos

## Práctica de laboratorio 01-Desarrollo e implementación de aplicaciones de computo paralelo basado en hilos y procesos

### Descripción de la Practica

Diseñar e implementar una aplicación usando los principales paradigmas de programación en paralelo como el procesamiento basado en hilos y procesos. La aplicación que se pide implementar pretende generar una bolsa de palabras que puede ser utilizada para análisis de sentimiento, recuperación de información, entre otras técnicas dentro del área de Procesamiento de Lenguaje Natural (PLN).

### Descripción del dataset
El conjunto de datos de reseñas de Amazon consta de reseñas de clientes de Amazon. Los datos abarcan un período de 18 años, incluidas aproximadamente 35 millones de reseñas hasta marzo de 2013. Las reseñas incluyen información sobre productos, usuarios, calificaciones y una reseña en texto sin formato. 
Para obtener más información, consulte el siguiente artículo: J. McAuley and J. Leskovec. Hidden factors and hidden topics: understanding rating dimensions with review text. RecSys, 2013.


### Procedimiento

-	Descargar el dataset amazon_review_full_csv.tar.gz desde el siguiente enlace: https://figshare.com/articles/dataset/Amazon_Reviews_Full/13232537/1
-	Utilizar el archivo train.csv
-	Realizar un algoritmo que permita generar una bolsa de palabras para cada reseña según el índice de clase (1 a 5) aplicando paralelismo basado en procesos e hilos.
-	Determinar cuáles son las palabras que más se repiten según cada índice de clase. 

### Presentación de resultados

-	Gráfica comparativa de los tiempos de ejecución entre un algoritmo secuencial y algoritmo paralelo

![24JPA](https://github.com/Juancarlos56/Practica01-Desarrollo-Computo-Paralelo-Con-Procesos/blob/main/imagenes/comparativa.png)



### Realizado por: 
- Juan Carlos Barrera B
- Katherine Michelle Barrera B.

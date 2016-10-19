# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

# Import data y el tensor
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

FLAGS = None


def main(_):
	
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  ''' La data esta dividida en 3 partes: 
	55K data points para el entrenamiento (mnist.train)	
	10K para el test (mnist.test)
	5K para la validacion (mnist.validation)
     Cada DATA POINT tiene dos partes:
	la imagen  ("X")  -> mnist.train.images
	la etiqueta correspondiente. ("Y")  -> mnist.train.labels
	    
     Cada imagen es de 28x28 pixeles (matriz) -> Lo podemos ver como un vector de
	    784 numeros cuando hacemos un Flatten. En el metodo de Softmax Regression no necesitamos explotar esa estructura 2D de la imagen.

     mnist.train.images es un tensor (un array n-dimensional) con forma [55k,784] 
	osea 55K imagenes donde cada una es un vector de 784 posiciones//pixeles. 
	     1er termino: indice en la lista de imagenes.
	     2o termino: indice a cada pixel de cada imagen.
	Cada entrada en el tensor es la intensidad de un pixel entre 0 y 1 para un pixel particular en una imagen particular. 
	
     Las etiquetas tendran un valor de 0 a 9 segun el numero escrito. En este caso usaremos one-hot vectors (todas la dimensiones a 0 excepto 1) para las etiquetas.
     mnist.train.labels tiene la forma [55K,10]

     Softmax regression es un modelo simple. Este nos da una lista de valores entre 0 y 1 que al sumarlos alcanzan el 1. Estos valores son las probabilidades de que la imagen pertenezca a una clase determinada. Se computa a partir de los scores. Para aplicar Softmax debemos realizar dos pasos:
         Sumamos todas las evidencias de que nuestra input sea de una determinadas clases. Lo haremos mediante pesos (Cuanto mayor sea-> mas a favor que la evidencia indique la clase acertada). 
	 Convertimos esa evidencia en probabilidades. 
	 Ademas anadimos un bias que nos da informacion extra independientemente de como sean las inputs que le pasemos.
	 y=nuestras probabilidades despues de aplicar softmax(evidencia)
	 Softmax nos sirve como funcion de activacion//enlace que da forma a la salida de tal manera que distribuye las probabilidades segun el numero de clases.

	y=softmax(Wx+B)
  '''

  # CREAMOS EL MODELO en TF. Al definir el modelo, ahora TF es capaz de entrenarlo facilmente porque sabe el grado entero de operaciones que vas a realizar. Entonces el automaticamente sabe aplicar el algoritmo de backpropagation para definir eficientemente como las variables (W y b) afectan a la perdida que deseamos minimizar. Podremos aplicar la optimizacion que deseemos

  # X es un placeholder, es decir, un valor que pondremos  como entrada cuando queramos que TF runee una operacion//calculo. Queremos poder meter como input cualquier numero de imagenes cada una de ellas representada con un vector de 784 posiciones. Por lo tanto definimos el placeholder con [None,784] donde el none implica que la dimension puede tener cualquier longitud.
  x = tf.placeholder(tf.float32, [None, 784]) 

  # Para los pesos de nuestro modelo creamos una Variable. Las variables son tensores modificables que se alojan en el grafo de interaccion de operaciones de TF. Como vemos se define como [784 pixeles,10 clases]. Es un tensor lleno de 0s
  W = tf.Variable(tf.zeros([784, 10]))
  # bias para las 10 clases. Es un tensor lleno de 0s
  b = tf.Variable(tf.zeros([10]))

  #Definimos el modelo. Las operaciones que lo definen. Luego ya aplicaremos softmax. y es la distribucion de probabilidad predicha.
  y = tf.matmul(x, W) + b

  # Defimos el coste e optimizacion. El coste nos indica cuan lejos esta nuestro modelo del deseado. y_ es la distribucion real (one-hot).
  y_ = tf.placeholder(tf.float32, [None, 10])

  
  ''' The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    outputs of 'y', and then average across the batch.
  '''

  #hacemos la entropia cruzada a nivel de logits.Respecto nuestra prediccion y la real.
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

  #Definimos que tipo de optimizacion queremos utilizar con tal de reducir la perdida. En este caso utilizamos el Gradiente Descendiente como optimizador para reducir la entropia cruzada con un rate de aprendizaje de 0.5
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()

  #Modelo configurado para el entrenamiento. Antes de lanzarlo creamos una operacion para poden inicializar todas las variables que hemos creado. y la runeamos.
 
 


# ENTRENAMIENTO
  tf.initialize_all_variables().run()

  #Hacemos la secuencia de entrenamiento 1K veces.
  for _ in range(1000):
  #En cada step del loop tenemos un batch de 100 data points de nuestro set de entrenamiento.
  #Usar pequenos batches de data random se le conoce como entrenamiento estocastico, en este caso  hacemos un gradiente descendiente estocastico. Idealmente nos gustaria trabajar con toda la informacion en casa step del entrenamiento pero eso es muy costoso.
    batch_xs, batch_ys = mnist.train.next_batch(100)
  #runeamos el train_step alimentando a x con una porcion de la info (batch_xs) y a las etiquetas con un pequeno batch_ys. los batches con tal que reeplacen a los placeholders.
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})




  # TESTEAMOS NUESTRO MODELO (ya entrenado)
  #tf.argmax(y,1) nos dara el indice del mayor valor dentro del tensor y en algunos ejes, es decir nos dara la etiqueta a la imagen que el modelo cree que es la correcta. En el caso de aplicar esa funcion con y_ nos devolvera la etitqueta real. Al hacer un equal obtendremos una lista de booleanos si nuestra prediccion coincide o no.
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

  #para saber la precision primero hacemos un cast es decir de [true,false,true,true] pasamos a [1,0,0,1]. Y luego le hacemos la media.
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print("precision")

  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))
  


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/data',
                      help='Directory for storing data')
  FLAGS = parser.parse_args()
  tf.app.run()

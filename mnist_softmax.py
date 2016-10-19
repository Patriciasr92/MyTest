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

  # CREAMOS EL MODELO en TF.
  # X es un placeholder, es decir, un valor que pondremos  como entrada cuando queramos que TF runee una operacion//calculo. Queremos poder meter como input cualquier numero de imagenes cada una de ellas representada con un vector de 784 posiciones. Por lo tanto definimos el placeholder con [None,784] donde el none implica que la dimension puede tener cualquier longitud.
  x = tf.placeholder(tf.float32, [None, 784]) 
  # Para los pesos de nuestro modelo creamos una Variable. Las variables son tensores modificables que se alojan en el grafo de interaccion de operaciones de TF. Como vemos se define como [784 pixeles,10 clases]. Es un tensor lleno de 0s
  W = tf.Variable(tf.zeros([784, 10]))
  # bias para las 10 clases. Es un tensor lleno de 0s
  b = tf.Variable(tf.zeros([10]))

  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  # Train
  tf.initialize_all_variables().run()
  for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
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

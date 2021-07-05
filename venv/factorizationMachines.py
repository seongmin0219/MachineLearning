

import numpy as np
import pandas as np
import tensorflow as tf
from tensorflow.keras.metrics import BinaryAccuracy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split



# GPU 확인
tf.config.list_physical_devies('GPU')

# 자료형 선언
tf.keras.backend.set_floatx('float32')

# 데이터 로드
scaler = MinMaxScaler()
file = load_breast_cancer()
X, Y = file['data'], file['target']
X = scaler.fit_transform(X)

n = X.shape[0]
p = X.shape[1]
k = 10
batch_size = 8
epochs = 10


class FM(tf.keras.Model):
    def __init__(self):
        super(FM, self).__init__()

        #모델의 파라미터 정의
        self.w_0 =tf.Variable([0.0])
        self.w = tf.Variable(tf.zeros([p]))
        self.V = tf.Variable(tf.random.normal(shape=(p,k)))

    def call(self, inputs):
        linear_terms = tf.reduce_sum(tf.math.multiply(self.w, inputs), axis=1)

        interactions = 0.5 * tf.reduce_sum(
            tf.math.pow(tf.matmul(inputs, self.V), 2)
            - tf.matmul(tf.math.pow(inputs, 2), tf.math.pow(self.V, 2)),
            1,
            keepdims=False
        )
        
        y_hat = tf.math.sigmoid(self.w_0 + linear_terms + interactions)

        return y_hat
import tensoflow as tf

class Attention(tf.keras.layers.Layer):
    def __init__(self, l2=0.0):
        super(Attention, self).__init__()

        self.l2 = l2

    def build(self, input_shape):
        self.W = self.add_weight(
          shape=(input_shape[1], self.units),
          initializer='glorot_uniform',
          regularizer=tf.keras.regularizers.l2(self.l2)
        )

        self.a_1 = self.add_weight(
          shape=(input_shape[1], self.units),
          initializer='glorot_uniform',
          regularizer=tf.keras.regularizers.l2(self.l2)
        )

        self.a_2 = self.add_weight(
          shape=(input_shape[1], self.units),
          initializer='glorot_uniform',
          regularizer=tf.keras.regularizers.l2(self.l2)
        )

    def call(self, H, A):

        X = H @ self.W

        attn_self = X @ self.a_1
        attn_neighbours = X @ self.a_2

        attention = attn_self + tf.transpose(attn_neighbours)

        E = tf.nn.leaky_relu(attention)
        A = tf.cast(tf.math.greater(A, 0.0), dtype=tf.float32)

        alpha = tf.nn.softmax(E * A)

        H_cap = alpha @ X



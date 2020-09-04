import tensoflow as tf

class Attention(tf.keras.layers.Layer):
    def __init__(self, units, activation=tf.identity, l2=0.0):
        super(Attention, self).__init__()

        self.l2 = l2
        self.activation = activation
        self.units = units

    def build(self, input_shape):

        H_shape, A_shape = input_shape

        self.W = self.add_weight(
          shape=(H_shape[1], self.units),
          initializer='glorot_uniform',
          regularizer=tf.keras.regularizers.l2(self.l2)
        )

        self.a_1 = self.add_weight(
          shape=(self.units, 1),
          initializer='glorot_uniform',
          regularizer=tf.keras.regularizers.l2(self.l2)
        )

        self.a_2 = self.add_weight(
          shape=(self.units, 1),
          initializer='glorot_uniform',
          regularizer=tf.keras.regularizers.l2(self.l2)
        )

    def call(self, inputs):

        H, A = inputs
        X = H @ self.W

        attn_self = X @ self.a_1
        attn_neighbours = X @ self.a_2

        attention = attn_self + tf.transpose(attn_neighbours)

        E = tf.nn.leaky_relu(attention)
        A = tf.cast(tf.math.greater(A, 0.0), dtype=tf.float32)

        alpha = tf.nn.softmax(E * A)

        H_cap = alpha @ X

        out = self.activation(H_cap)

        return out



class GraphAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units, num_heads, activation=tf.identity, l2=0.0):
        super(GraphAttentionLayer, self).__init__()

        self.activation = activation
        self.num_heads = num_heads

        self.attn_layers = [Attention(units, tf.identity, l2) for x in range(num_layers)]

    def call(self, inputs):

        H, A = inputs

        H_out = [self.attn_layers[i]([H, A]) for i in range(self.num_heads)]

        multi_head_attn = tf.concat(H_out, axis=-1)

        out = self.activation(multi_head_attn)

        return out

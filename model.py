import tensorflow as tf
from tensorflow import keras
from proteinbert.model_generation import ModelGenerator


class BahdanauAttention(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
        self.units = units
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def get_config(self):
        base_config = super(BahdanauAttention, self).get_config()
        base_config['units'] = self.units
        return base_config

    @classmethod
    def from_config(cls, config):
        print("Config received", config)
        return cls(**config)

    def call(self, query, values, mask=None,**kwargs):
        hidden_with_time_axis = tf.expand_dims(query, 1)

        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        if mask is not None:
            mask = tf.expand_dims(mask, -1)
            score += (1.0 - tf.cast(mask, dtype=score.dtype)) * -1e9

        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


class AttentionBasedModelGenerator(ModelGenerator):
    def __init__(self, pretraining_model_generator, output_spec, pretraining_model_manipulation_function=None,
                 dropout_rate=0.5, optimizer_class=None, lr=None, other_optimizer_kwargs=None, model_weights=None,
                 optimizer_weights=None):

        if other_optimizer_kwargs is None:
            if optimizer_class is None:
                other_optimizer_kwargs = pretraining_model_generator.other_optimizer_kwargs
            else:
                other_optimizer_kwargs = {}

        if optimizer_class is None:
            optimizer_class = pretraining_model_generator.optimizer_class

        if lr is None:
            lr = pretraining_model_generator.lr

        ModelGenerator.__init__(self, optimizer_class=optimizer_class, lr=lr,
                                other_optimizer_kwargs=other_optimizer_kwargs, model_weights=model_weights, \
                                optimizer_weights=optimizer_weights)

        self.pretraining_model_generator = pretraining_model_generator
        self.output_spec = output_spec
        self.pretraining_model_manipulation_function = pretraining_model_manipulation_function
        self.dropout_rate = dropout_rate

    def create_model(self, seq_len, freeze_pretrained_layers=False):

        model = self.pretraining_model_generator.create_model(seq_len, compile=False,
                                                              init_weights=(self.model_weights is None))

        if self.pretraining_model_manipulation_function is not None:
            model = self.pretraining_model_manipulation_function(model)

        if freeze_pretrained_layers:
            for layer in model.layers:
                layer.trainable = False

        model_inputs = model.input
        pretraining_output_seq_layer, _ = model.output
        last_hidden_layer = pretraining_output_seq_layer

        masks = model_inputs[-1]  # [batch_size, seq_len]
        dropout = keras.layers.Dropout(self.dropout_rate)(last_hidden_layer)

        # Attention layer
        attention_layer = BahdanauAttention(units=128)
        attention_outputs, _ = attention_layer(dropout, dropout, masks)
        con_outputs = keras.layers.Concatenate()([dropout, attention_outputs])
        dropout = keras.layers.Dropout(self.dropout_rate)(con_outputs)

        # Bi-GRU Layer
        bigru = keras.layers.Bidirectional(keras.layers.GRU(units=128, return_sequences=True))(dropout)
        dropout = keras.layers.Dropout(self.dropout_rate)(bigru)

        # Output Layer
        output_layer = keras.layers.Dense(1, activation='sigmoid')(dropout)

        loss = 'binary_crossentropy'
        model = keras.models.Model(inputs=model_inputs, outputs=output_layer)
        model.compile(loss=loss, optimizer=self.optimizer_class(learning_rate=self.lr, **self.other_optimizer_kwargs))
        self._init_weights(model)

        return model


class CNNBasedModelGenerator(ModelGenerator):
    def __init__(self, pretraining_model_generator, output_spec, pretraining_model_manipulation_function=None,
                 dropout_rate=0.5, optimizer_class=None, \
                 lr=None, other_optimizer_kwargs=None, model_weights=None, optimizer_weights=None):

        if other_optimizer_kwargs is None:
            if optimizer_class is None:
                other_optimizer_kwargs = pretraining_model_generator.other_optimizer_kwargs
            else:
                other_optimizer_kwargs = {}

        if optimizer_class is None:
            optimizer_class = pretraining_model_generator.optimizer_class

        if lr is None:
            lr = pretraining_model_generator.lr

        ModelGenerator.__init__(self, optimizer_class=optimizer_class, lr=lr,
                                other_optimizer_kwargs=other_optimizer_kwargs, model_weights=model_weights, \
                                optimizer_weights=optimizer_weights)

        self.pretraining_model_generator = pretraining_model_generator
        self.output_spec = output_spec
        self.pretraining_model_manipulation_function = pretraining_model_manipulation_function
        self.dropout_rate = dropout_rate

    def create_model(self, seq_len, freeze_pretrained_layers=False):

        model = self.pretraining_model_generator.create_model(seq_len, compile=False,
                                                              init_weights=(self.model_weights is None))

        if self.pretraining_model_manipulation_function is not None:
            model = self.pretraining_model_manipulation_function(model)

        if freeze_pretrained_layers:
            for layer in model.layers:
                layer.trainable = False

        model_inputs = model.input
        pretraining_output_seq_layer, _ = model.output
        last_hidden_layer = pretraining_output_seq_layer
        # Bi-GRU layer
        dropout = keras.layers.Dropout(self.dropout_rate)(last_hidden_layer)
        bigru = keras.layers.Bidirectional(keras.layers.GRU(units=128, return_sequences=True))(dropout)

        # CNN layers
        dropout = keras.layers.Dropout(self.dropout_rate)(bigru)
        cnn_outputs = keras.layers.Conv1D(filters=256, kernel_size=3, strides=1, padding='same')(dropout)
        norm_outputs = keras.layers.BatchNormalization()(cnn_outputs)
        res_outputs = keras.layers.Add()([norm_outputs, dropout])
        dropout = keras.layers.Dropout(self.dropout_rate)(res_outputs)

        cnn_outputs = keras.layers.Conv1D(filters=256, kernel_size=5, strides=1, padding='same')(dropout)
        norm_outputs = keras.layers.BatchNormalization()(cnn_outputs)
        res_outputs = keras.layers.Add()([norm_outputs, dropout])
        dropout = keras.layers.Dropout(self.dropout_rate)(res_outputs)

        # Bi-GRU layer
        bigru = keras.layers.Bidirectional(keras.layers.GRU(units=128, return_sequences=True))(dropout)
        dropout = keras.layers.Dropout(self.dropout_rate)(bigru)

        output_layer = keras.layers.Dense(1, activation='sigmoid')(dropout)

        loss = 'binary_crossentropy'

        model = keras.models.Model(inputs=model_inputs, outputs=output_layer)
        model.compile(loss=loss, optimizer=self.optimizer_class(learning_rate=self.lr, **self.other_optimizer_kwargs))

        self._init_weights(model)

        return model
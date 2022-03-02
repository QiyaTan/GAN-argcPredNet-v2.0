import numpy as np

from keras import backend as K
from keras import activations
from keras.layers import Recurrent
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D
from keras.engine import InputSpec
class Argc_PredNet(Recurrent):

    def __init__(self, stack_sizes, R_stack_sizes,
                 A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                 pixel_max=1., error_activation='relu', A_activation='relu',
                 LSTM_activation='tanh', LSTM_inner_activation='hard_sigmoid',
                 output_mode='error', extrap_start_time=None,
                 data_format=K.image_data_format(), **kwargs):
        self.stack_sizes = stack_sizes
        self.nb_layers = len(stack_sizes)
        assert len(R_stack_sizes) == self.nb_layers, 'len(R_stack_sizes) must equal len(stack_sizes)'  # assert语句，用于确保R部层数与A部一样
        self.R_stack_sizes = R_stack_sizes
        assert len(A_filt_sizes) == (self.nb_layers - 1), 'len(A_filt_sizes) must equal len(stack_sizes) - 1'
        self.A_filt_sizes = A_filt_sizes
        assert len(Ahat_filt_sizes) == self.nb_layers, 'len(Ahat_filt_sizes) must equal len(stack_sizes)'
        self.Ahat_filt_sizes = Ahat_filt_sizes
        assert len(R_filt_sizes) == (self.nb_layers), 'len(R_filt_sizes) must equal len(stack_sizes)'
        self.R_filt_sizes = R_filt_sizes

        self.pixel_max = pixel_max
        self.error_activation = activations.get(error_activation)
        self.A_activation = activations.get(A_activation)
        self.LSTM_activation = activations.get(LSTM_activation)
        self.LSTM_inner_activation = activations.get(LSTM_inner_activation)

        default_output_modes = ['prediction', 'error', 'all']
        layer_output_modes = [layer + str(n) for n in range(self.nb_layers) for layer in ['R', 'E', 'A', 'Ahat']]
        assert output_mode in default_output_modes + layer_output_modes, 'Invalid output_mode: ' + str(output_mode)
        self.output_mode = output_mode
        if self.output_mode in layer_output_modes:
            self.output_layer_type = self.output_mode[:-1]
            self.output_layer_num = int(self.output_mode[-1])
        else:
            self.output_layer_type = None
            self.output_layer_num = None
        self.extrap_start_time = extrap_start_time

        assert data_format in {'channels_last', 'channels_first'}, 'data_format must be in {channels_last, channels_first}'
        self.data_format = data_format
        self.channel_axis = -3 if data_format == 'channels_first' else -1
        self.row_axis = -2 if data_format == 'channels_first' else -3
        self.column_axis = -1 if data_format == 'channels_first' else -2

        super(Argc_PredNet, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=5)]


    def compute_output_shape(self, input_shape):
        if self.output_mode == 'prediction':
            out_shape = input_shape[2:]
        elif self.output_mode == 'error':
            out_shape = (self.nb_layers,)
        elif self.output_mode == 'all':
            out_shape = (np.prod(input_shape[2:]) + self.nb_layers,)
        else:
            stack_str = 'R_stack_sizes' if self.output_layer_type == 'R' else 'stack_sizes'
            stack_mult = 2 if self.output_layer_type == 'E' else 1
            out_stack_size = stack_mult * getattr(self, stack_str)[self.output_layer_num]
            out_nb_row = input_shape[self.row_axis] / 2**self.output_layer_num
            out_nb_col = input_shape[self.column_axis] / 2**self.output_layer_num
            if self.data_format == 'channels_first':
                out_shape = (out_stack_size, out_nb_row, out_nb_col)
            else:
                out_shape = (out_nb_row, out_nb_col, out_stack_size)

        if self.return_sequences:
            return (input_shape[0], input_shape[1]) + out_shape
        else:
            return (input_shape[0],) + out_shape

    def get_initial_state(self, x):
        input_shape = self.input_spec[0].shape
        init_nb_row = input_shape[self.row_axis]
        init_nb_col = input_shape[self.column_axis]

        base_initial_state = K.zeros_like(x)
        non_channel_axis = -1 if self.data_format == 'channels_first' else -2
        for _ in range(2):
            base_initial_state = K.sum(base_initial_state, axis=non_channel_axis)
        base_initial_state = K.sum(base_initial_state, axis=1)

        initial_states = []
        states_to_pass = ['r', 'c', 'e']
        nlayers_to_pass = {u: self.nb_layers for u in states_to_pass}
        if self.extrap_start_time is not None:
           states_to_pass.append('ahat')
           nlayers_to_pass['ahat'] = 1
        for u in states_to_pass:
            for l in range(nlayers_to_pass[u]):
                ds_factor = 2 ** l
                nb_row = init_nb_row // ds_factor
                nb_col = init_nb_col // ds_factor
                if u in ['r', 'c']:
                    stack_size = self.R_stack_sizes[l]
                elif u == 'e':
                    stack_size = 2 * self.stack_sizes[l]
                elif u == 'ahat':
                    stack_size = self.stack_sizes[l]
                output_size = stack_size * nb_row * nb_col

                reducer = K.zeros((input_shape[self.channel_axis], output_size))
                initial_state = K.dot(base_initial_state, reducer)
                if self.data_format == 'channels_first':
                    output_shp = (-1, stack_size, nb_row, nb_col)
                else:
                    output_shp = (-1, nb_row, nb_col, stack_size)
                initial_state = K.reshape(initial_state, output_shp)
                initial_states += [initial_state]

        #if K._BACKEND == 'theano':
         #   from theano import tensor as T
            # There is a known issue in the Theano scan op when dealing with inputs whose shape is 1 along a dimension.
            # In our case, this is a problem when training on grayscale images, and the below line fixes it.
         #   initial_states = [T.unbroadcast(init_state, 0, 1) for init_state in initial_states]

        if self.extrap_start_time is not None:
            initial_states += [K.variable(0, int if K.backend() != 'tensorflow' else 'int32')]
        return initial_states

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.conv_layers = {c: [] for c in ['i', 'f', 'c', 'a', 'ahat']}

        for l in range(self.nb_layers):

            for c in ['i', 'f', 'c']:
                act = self.LSTM_activation if c == 'c' else self.LSTM_inner_activation

                self.conv_layers[c].append(Conv2D(self.R_stack_sizes[l], self.R_filt_sizes[l], padding='same', activation=act, data_format=self.data_format))

            act = 'relu' if l == 0 else self.A_activation
            self.conv_layers['ahat'].append(Conv2D(self.stack_sizes[l], self.Ahat_filt_sizes[l], padding='same', activation=act, data_format=self.data_format))

            if l < self.nb_layers - 1:
                self.conv_layers['a'].append(Conv2D(self.stack_sizes[l+1], self.A_filt_sizes[l], padding='same', activation=self.A_activation, data_format=self.data_format))

        self.upsample = UpSampling2D(data_format=self.data_format)
        self.pool = MaxPooling2D(data_format=self.data_format)

        self.trainable_weights = []
        nb_row, nb_col = (input_shape[-2], input_shape[-1]) if self.data_format == 'channels_first' else (input_shape[-3], input_shape[-2])
        for c in sorted(self.conv_layers.keys()):
            for l in range(len(self.conv_layers[c])):
                ds_factor = 2 ** l
                if c == 'ahat':
                    nb_channels = self.R_stack_sizes[l]
                elif c == 'a':
                    nb_channels = 2 * self.stack_sizes[l]
                else:
                    nb_channels = self.stack_sizes[l] * 2 + self.R_stack_sizes[l]
                    if l < self.nb_layers - 1:
                        nb_channels += self.R_stack_sizes[l+1]
                in_shape = (input_shape[0], nb_channels, nb_row // ds_factor, nb_col // ds_factor)
                if self.data_format == 'channels_last': in_shape = (in_shape[0], in_shape[2], in_shape[3], in_shape[1])
                with K.name_scope('layer_' + c + '_' + str(l)):
                    self.conv_layers[c][l].build(in_shape)
                self.trainable_weights += self.conv_layers[c][l].trainable_weights

#        if self._initial_weights is not None:          # keras2.0中改成_initial_weights
 #           self.set_weights(self._initial_weights)
 #           del self._initial_weights

        self.states = [None] * self.nb_layers*3

        if self.extrap_start_time is not None:
            self.t_extrap = K.variable(self.extrap_start_time, int if K.backend() != 'tensorflow' else 'int32')
            self.states += [None] * 2

    def step(self, a, states):
        r_tm1 = states[:self.nb_layers]
        c_tm1 = states[self.nb_layers:2*self.nb_layers]
        e_tm1 = states[2*self.nb_layers:3*self.nb_layers]

        if self.extrap_start_time is not None:
            t = states[-1]
            a = K.switch(t >= self.t_extrap, states[-2], a)

        c = []
        r = []
        e = []
        # R Unit
        for l in reversed(range(self.nb_layers)):
            inputs = [r_tm1[l], e_tm1[l]]
            if l < self.nb_layers - 1:

                inputs.append(r_up)

            inputs = K.concatenate(inputs, axis=self.channel_axis)
            i = self.conv_layers['i'][l].call(inputs)
            f = self.conv_layers['f'][l].call(inputs)
            # o = self.conv_layers['o'][l].call(inputs)
            _c = f * c_tm1[l] + i * self.conv_layers['c'][l].call(inputs)
            _r = i * self.LSTM_activation(_c)
            c.insert(0, _c)
            r.insert(0, _r)

            if l > 0:
                r_up = self.upsample.call(_r)

        for l in range(self.nb_layers):
            ahat = self.conv_layers['ahat'][l].call(r[l])
            if l == 0:
                ahat = K.minimum(ahat, self.pixel_max)
                frame_prediction = ahat

            # compute errors
            e_up = self.error_activation(ahat - a)
            e_down = self.error_activation(a - ahat)

            e.append(K.concatenate((e_up, e_down), axis=self.channel_axis))
            if self.output_layer_num == l:
                if self.output_layer_type == 'A':
                    output = a
                elif self.output_layer_type == 'Ahat':
                    output = ahat
                elif self.output_layer_type == 'R':
                    output = r[l]
                elif self.output_layer_type == 'E':
                    output = e[l]

            if l < self.nb_layers - 1:
                a = self.conv_layers['a'][l].call(e[l])
                a = self.pool.call(a)

        if self.output_layer_type is None:
            if self.output_mode == 'prediction':
                output = frame_prediction
            else:
                for l in range(self.nb_layers):
                    layer_error = K.mean(K.batch_flatten(e[l]), axis=-1, keepdims=True)
                    all_error = layer_error if l == 0 else K.concatenate((all_error, layer_error), axis=-1)
                if self.output_mode == 'error':
                    output = all_error
                else:
                    output = K.concatenate((K.batch_flatten(frame_prediction), all_error), axis=-1)

        states = r + c + e
        if self.extrap_start_time is not None:
            states += [frame_prediction, t + 1]

        return output, states

    def get_config(self):
        config = {'stack_sizes': self.stack_sizes,
                  'R_stack_sizes': self.R_stack_sizes,
                  'A_filt_sizes': self.A_filt_sizes,
                  'Ahat_filt_sizes': self.Ahat_filt_sizes,
                  'R_filt_sizes': self.R_filt_sizes,
                  'pixel_max': self.pixel_max,
                  'error_activation': self.error_activation.__name__,
                  'A_activation': self.A_activation.__name__,
                  'LSTM_activation': self.LSTM_activation.__name__,
                  'LSTM_inner_activation': self.LSTM_inner_activation.__name__,
                  'data_format': self.data_format,
                  'extrap_start_time': self.extrap_start_time,
                  'output_mode': self.output_mode}
        base_config = super(Argc_PredNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

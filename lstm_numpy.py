from turtle import forward
import numpy as np

# activation function

def tanh_func(x):
    return np.tanh(x)

def sigmoid_func(x):
    return 1/(1+np.exp(-x))

def softmax_func(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# weight initialize

def weight_init(shape, n):
    return np.random.randn(shape[0], shape[1]) / np.sqrt(1/n)

def lstm_weight_init(net_size, embedded_size):
    
    element = ['computing', 'update', 'forget', 'output']
    W = {}
    b = {}
    n = net_size + embedded_size
    for e in element:
        W[f'W{e}'] = weight_init((net_size, n), n)
        b[f'b{e}'] = weight_init((net_size,), n)
    return W, b

# Backbone

class LSTM:
    def __init__(self, embedded_size, net_size):
        self.embedded_size = embedded_size
        self.net_size = net_size
        self.W, self.b = lstm_weight_init(net_size, embedded_size)
    
    def _linear_compute(self, a_x: np.array, element_role: str):
        return np.matmul(self.W[f'W{element_role}'], a_x) \
                        + self.b[f'W{element_role}']
    
    def _gated_compute(self, a_x, element_role):
        return sigmoid_func(self._linear_compute(a_x, element_role))
    
    def lstm_cell(self, a, x_t):
        a_x = np.concatenate((a, x_t), axis=1)
        c_tilde = tanh_func(self._linear_compute(a_x, 'computing'))
        update_gate = self._gated_compute(a_x, 'update')
        forget_gate = self._gated_compute(a_x, 'forget')
        output_gate = self. _gated_compute(a_x, 'output')
        c = update_gate * c_tilde + forget_gate * c
        a = output_gate * tanh_func(c)

    def forward(self, x):
        # write reasonable forward method for your problems
        return x
##

 
     
        

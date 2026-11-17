
import numpy as np
class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the RNN with random weights and zero biases.
        """
        self.hidden_size = hidden_size
        self.W_xh = np.random.randn(hidden_size, input_size)*0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size)*0.01
        self.W_hy = np.random.randn(output_size, hidden_size)*0.01
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))

    def forward(self, x):
        """
        Forward pass through the RNN for a given sequence of inputs.
        """
        # initial hidden state = 0
        h = np.zeros((self.hidden_size, 1))
        self.last_inputs = []
        self.last_hiddens = [h]
        outputs = []
        # for each time step 't'
        for x_t in x:
            x_t = np.reshape(x_t, (-1,1))   # (input_size, 1)
            h = np.tanh(self.W_xh @ x_t + self.W_hh @ h + self.b_h)
            y_t = self.W_hy @ h + self.b_y

            self.last_inputs.append(x_t)
            self.last_hiddens.append(h)
            outputs.append(y_t)

        return np.array(outputs) # (T, output_size, 1)

    def backward(self, x, y, learning_rate):
        """
        Backpropagation through time to adjust weights based on error gradient.
        """
        outputs = self.forward(x)
        T = len(self.last_inputs)

        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)

        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        dh_next = np.zeros((self.hidden_size,1)) # gradients from step t+1


        for t in reversed(range(T)):
            y_t = np.reshape(y[t], (-1,1))
            # dL / dy_t = y_hat - y_t
            dy = outputs[t] - y_t

            h_t = self.last_hiddens[t+1] # fetch current hidden state for timestep 't' 
            h_prev = self.last_hiddens[t]
            x_t = self.last_inputs[t]

            # gradients for the output layer weights and biases:
            dW_hy += dy @ h_t.T 
            db_y += dy
            dh = self.W_hy.T @ dy + dh_next
            dh_raw = dh * (1-h_t**2)
            dh_next = self.W_hh.T @ dh_raw

            dW_xh += dh_raw @ x_t.T
            dW_hh += dh_raw @ h_prev.T
            db_h += dh_raw

        self.W_xh -= learning_rate * dW_xh
        self.W_hh -= learning_rate * dW_hh
        self.W_hy -= learning_rate * dW_hy
        self.b_h -= learning_rate * db_h
        self.b_y -= learning_rate * db_y






        

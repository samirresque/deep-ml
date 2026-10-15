import numpy as np

def sigmoid(x):
	return 1/(1+np.exp(-x))

class LSTM:
	def __init__(self, input_size, hidden_size):
		self.input_size = input_size
		self.hidden_size = hidden_size

		# Initialize weights and biases
		self.Wf = np.random.randn(hidden_size, input_size + hidden_size) # forget gate
		self.Wi = np.random.randn(hidden_size, input_size + hidden_size) # input gate
		self.Wc = np.random.randn(hidden_size, input_size + hidden_size) # candidate cell state
		self.Wo = np.random.randn(hidden_size, input_size + hidden_size) # output state

		self.bf = np.zeros((hidden_size, 1))
		self.bi = np.zeros((hidden_size, 1))
		self.bc = np.zeros((hidden_size, 1))
		self.bo = np.zeros((hidden_size, 1))

	def forward(self, x, initial_hidden_state, initial_cell_state):
		"""
		Processes a sequence of inputs and returns the hidden states, final hidden state, and final cell state.
		"""
		h_t = initial_hidden_state
		c_t = initial_cell_state
		hidden_states = []
		for x_t in x:
			x_t = x_t.reshape(-1,1)
			h_t = h_t.reshape(-1,1)
			u_t = np.concatenate((h_t, x_t), axis=0)
			# u_t = np.transpose(u_t)
			f_t = sigmoid(np.dot(self.Wf, u_t) + self.bf) # forget gate: decides which information to discard from the cell hidden_states

			i_t = sigmoid(np.dot(self.Wi, u_t)+ self.bi) # input gate: decides what to update
			c_tilde = np.tanh(np.dot(self.Wc, u_t)+ self.bc) # candidate cell: a vector of new candidate values

			c_t = f_t*c_t + i_t*c_tilde # cell state: combines prev c_t with c_tildes

			o_t = sigmoid(np.dot(self.Wo,u_t) + self.bo) # output state: what should be the next hidden state !

			h_t = o_t*np.tanh(c_t)
			hidden_states.append(h_t)
		
		return hidden_states, h_t, c_t 



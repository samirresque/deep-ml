import numpy as np

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:

	input_sequence = np.array(input_sequence)
	initial_hidden_state = np.array(initial_hidden_state)
	Wx = np.array(Wx)
	Wh = np.array(Wh)
	b = np.array(b)
	previous_hidden_state = initial_hidden_state
	for sequence in input_sequence: 
		next_hidden_state = np.tanh(Wx@sequence + Wh@previous_hidden_state + b)
		previous_hidden_state = next_hidden_state
		
	final_hidden_state = next_hidden_state
	return final_hidden_state
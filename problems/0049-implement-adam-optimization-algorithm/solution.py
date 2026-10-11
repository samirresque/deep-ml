import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
    # Note: `f` is accepted only for interface parity with the PyTorch/Tinygrad
    # variants, which derive gradients from it via autograd. This version uses
    # `grad` only — the objective value `f` is never evaluated.
    # 1. initialize moments: first and second
    m_t = 0
    v_t = 0
    theta = x0 # parameters
    for t in range(1,num_iterations+1):
        g_t = grad(theta) # gradients
        m_t = beta1*m_t + (1-beta1)*g_t # first moment
        v_t = beta2*v_t + (1-beta2)*(g_t**2)

        # bias correction
        m_hat = m_t/(1-beta1**t)
        v_hat = v_t/(1-beta2**t)
        theta = theta - learning_rate*m_hat/(np.sqrt(v_hat)+epsilon)
    
    return theta



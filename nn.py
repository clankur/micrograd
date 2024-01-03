from engine import Value
import random
from typing import List

class Neuron:
    """

    """
    def __init__(self, nin) -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x) -> Value:
        act = sum ([wi*xi for wi, xi in zip(self.w, x)], self.b) 
        out = act.tanh()
        return out
    
    def parameters(self) -> List[Value]:
        return self.w + [self.b]
 
class Layer:
    """

    """
    def __init__(self, nin, nout) -> None:
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x) -> List[Value]: 
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self) -> List[Value]:
        return [p for n in self.neurons for p in n.parameters()]
    
class MLP:
    """
    Multi-layer perceptron (MLP)
    """
    def __init__(self, nin: int, nouts: List[int]) -> None:
        szs = [nin] + nouts
        self.layers = [Layer(szs[i], szs[i+1]) for i in range(len(szs) - 1)]
    
    def __call__(self, x) -> Value:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Value]:
        return [p for layer in self.layers for p in layer.parameters()]
 
if __name__ == "__main__":
    n = MLP(3, [4, 4, 1])

    # writing a basic binary classifier neural network
    xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0] # desired targets
    ypred = [n(x) for x in xs]
    ypred

    # training
    for k in range(20):
        # forward pass
        ypred = [n(x) for x in xs]
        loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)])

        # need to reset the grads to 0 before backward pass
        for p in n.parameters():
            p.grad = 0.0
        # backward pass
        loss.backward()

        # update: perform gradient descent on the parameters
        for p in n.parameters():
            p.data += -0.05 * p.grad
        
        print(k, loss.data)
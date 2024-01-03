from numbers import Number
from typing import Tuple
from draw_helper import draw_dot
import math

class Value:
    def __init__ (self, data: Number, _children:Tuple=(), _op:str='', label='') -> None:
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other) -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __mul__(self, other) -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        out =  Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
 
    def __pow__(self, other) -> 'Value':
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
          
        return out
      
    def __rmul__(self, other) -> 'Value':
        return self * other

    def __truediv__(self, other) -> 'Value':
        return self * other**-1
 
    def __neg__ (self) -> 'Value':
        return self * -1
    
    def __sub__(self, other) -> 'Value':
        return self + (-other)

    def __radd__(self, other) -> 'Value':
        return self + other
    
   
    def tanh (self) -> 'Value':
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')
        
        def _backward():
            self.grad = (1 - t**2)* out.grad
        out._backward = _backward
        
        return out

    def exp (self) -> 'Value':

        x = self.data
        out = Value(math.exp(x), (self,), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        
        return out

    def backward (self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
            
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

if __name__ == "__main__":
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a * b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = Value(-2.0, label='f')
    L = d * f; L.label = 'L'
    L.backward()
    draw_dot(L)

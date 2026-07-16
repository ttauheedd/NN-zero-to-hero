#!/usr/bin/env python
# coding: utf-8

# In[1]:

import math

class Value:
    def __init__(self, data, _children=(), _op=()):
        self.data=data
        self.grad= 0
        self._backward=  lambda: None
        self._prev= set(_children)
        self._op= _op

    def __add__(self,other):
        other= other if isinstance(other, Value) else Value(other)
        out= Value(self.data + other.data, (self,other), '+')

        def _backward():
            self.grad+= out.grad
            other.grad+=out.grad
        out._backward=_backward

        return out

    def __mul__(self,other):
        other= other if isinstance(other, Value) else Value(other)
        out= Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad+= other.data * out.grad
            other.grad+= self.data *out.grad
        out._backward= _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out


    def log(self):
        # A tiny epsilon to prevent log(0) crashes
        epsilon = 1e-6
        
        # If the number is smaller than epsilon (like 0.0), just use epsilon
        safe_data = self.data if self.data > epsilon else epsilon
        
        out = Value(math.log(safe_data), (self,), 'log')

        def _backward():
            self.grad += (1.0 / safe_data) * out.grad
        out._backward = _backward

        return out

    def tanh(self) :
        e= math.exp(2*self.data)
        t= (e-1)/(e+1)
        out= Value(t, (self, ), 'tanh')

        def _backward():
            self.grad+= (1- t**2) * out.grad

        out._backward= _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def sigmoid(self):
        if self.data >= 0:
            res = 1 / (1 + math.exp(-self.data))
        else:
            e_x = math.exp(self.data)
            res = e_x / (1 + e_x)
            
        out = Value(res, (self,), 'sigm')

        def _backward():
            self.grad += (res * (1 - res)) * out.grad
        out._backward = _backward

        return out


    def backward(self):

        # topological order 
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()


    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"




# In[2]:


import random


# In[3]:


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad=0

    def parameters(self):
        return []


# In[4]:


class Neuron(Module):

    def __init__(self,nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b= Value(0)
        self.nonlin= nonlin

    def __call__(self, x):
        act= sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.sigmoid() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


# In[5]:


class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons= [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out= [n(x) for n in self.neurons]
        return out[0] if len(out)==1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', ' .join(str(n) for n in self.neurons)}]"


# In[6]:


class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"


# In[ ]:





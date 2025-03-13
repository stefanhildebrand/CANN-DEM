import torch

class Quadratic(torch.nn.Module):
    def __init__(self):
        super(Quadratic, self).__init__()
        return
    def forward(self, x):
        return x*x
    
class QuadraticLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QuadraticLayer, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.quadratic = Quadratic()

    def forward(self, x):
        x = self.linear(x)
        return self.quadratic(x)

class Inverse(torch.nn.Module):
    def __init__(self):
        super(Inverse, self).__init__()
        return
    def forward(self, x):
        return 1/x
    
class Cubic(torch.nn.Module):
    def __init__(self):
        super(Cubic, self).__init__()
        return
    def forward(self, x):
        return x*x*x
    
class Exponential(torch.nn.Module):
    def __init__(self):
        super(Exponential, self).__init__()
        return
    def forward(self, x):
        exp_x = torch.exp(x)-1
        clamped_output = torch.clamp(exp_x, max=1e2) 
        return clamped_output
    
'''    
class Logarithmic(torch.nn.Module):
    def __init__(self, learnable_parameter=1.0):
        super(Logarithmic, self).__init__()
        self.learnable_parameter = torch.nn.Parameter(torch.tensor(learnable_parameter))
        return
    def forward(self, x):
        log_x = torch.where(x > 0, torch.log(x + 1), self.learnable_parameter * x)
        return log_x
'''
    
class Logarithmic(torch.nn.Module):
    def __init__(self):
        super(Logarithmic, self).__init__()
        return
    def forward(self, x):
        log_x = torch.log(1-x)
        return log_x
import torch
import torch.nn as nn
from activations import *

# ---------------------------------------------------------------------------------------
# Neural Network
# ---------------------------------------------------------------------------------------

class CANN_2D(nn.Module):
    def __init__(self):
        super(CANN_2D, self).__init__()
        nodes_hl = 12

        # 2nd hidden layer (13 nodes)
        self.hidden_layer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1,1,bias=False),
                Exponential()
            ) for _ in range(nodes_hl//2)
        ])

        # Output Layer (1 node)
        self.output_layer = nn.Linear(nodes_hl,1, bias=False)
    
    def forward(self, F):
        
        # clamp parameters without loop
        if False:
            parameters = torch.nn.utils.parameters_to_vector(self.parameters())
            parameters[parameters < max(parameters)*1e-3] = 0
            torch.nn.utils.vector_to_parameters(parameters, self.parameters())
        
        # calculate C, Invariants, alternative Invariants, J
        C = torch.matmul(F.transpose(-1,-2),F) # C = F^T * F
        I1 = torch.einsum('ijj -> i', C) # I1 = tr(C)
        I2 = 0.5 * (I1**2 - (C * C).sum(dim=(-1, -2))) # I2 = 1/2 *(I1^2 - C:C)
        I3 = torch.det(C)
        J = torch.det(F)
        I1_ = I1/(J**(2/3))
        I2_ = I2/(J**(3/4))

        # extend by one dimension to match tensor of F's
        I1 = I1.unsqueeze(dim=0)
        I2 = I2.unsqueeze(dim=0)
        I1_ = I1_.unsqueeze(dim=0)
        I2_ = I2_.unsqueeze(dim=0)
        J = J.unsqueeze(dim=0)

        # calculate path through nodes
        out = torch.cat((I1-2, I2-2, J-1), dim = 0).T
        out = torch.stack((out, out**2)).transpose(0,2).transpose(0,1).reshape(-1,6) # 6 Nodes
        out_hl = torch.cat([node(out.detach()[:,i:i+1]) for i, node in enumerate(self.hidden_layer)], dim=1)
        # alternating out and out_hl elements
        out = torch.stack((out, out_hl)).transpose(0,2).transpose(0,1).reshape(-1,12) # 12 Nodes
        Psi = self.output_layer(out)

        # output P as gradient of Psi
        if not self.training:
            P = torch.autograd.grad(Psi, F, grad_outputs=torch.ones_like(Psi),
                                    create_graph=True, retain_graph=True)[0]   
            return Psi, P
        else:
            return Psi

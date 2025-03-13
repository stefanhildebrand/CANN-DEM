import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchopt

from activations import *
from loss_plot import LossPlot, ParamLossPlot
from DEM_2D import DeepEnergyMethod
from plot_beam import BeamPlotter
from DeepEnergyMethod.dem_hyperelasticity.Beam2D import config as cf
from DeepEnergyMethod.dem_hyperelasticity.Beam2D import define_structure as des

torch.set_default_device(torch.device("cuda"))
device = torch.device("cuda")

class CANN(nn.Module):
    # this NN is used to approximate the energy density psi
    # F -> psi
    def __init__(self):
        super(CANN, self).__init__()

        self.output_layer = nn.Linear(8, 1, bias=False)
        self.w = nn.Parameter(torch.ones(4), requires_grad=True)

        torch.nn.init.normal_(self.output_layer.weight, mean=10, std=0.0)

    def forward(self, F):

        # clamp parameters without loop
        parameters = torch.nn.utils.parameters_to_vector(self.parameters())
        parameters[parameters < max(parameters)*1e-3] = 0
        torch.nn.utils.vector_to_parameters(parameters, self.parameters())

        C = torch.matmul(F.transpose(-1,-2),F) # C = F^T * F
        I1 = torch.einsum('ijj -> i', C) # I1 = tr(C)
        J = torch.det(F)

        out = torch.stack((I1-2, J-1), dim=1)
        out = torch.cat((out, out**2), dim=1)

        #out = torch.stack((I1-2, J-1, (I1-2)**2, (J-1)**2), dim=1)

        out_exp = torch.exp(0.01*self.w * out)-1
        out_exp = torch.clamp(out_exp, max=1e2)
        #out_ln = torch.log(1-0.001*self.v * out)
        out = torch.cat((out, out_exp), dim=1)

        psi = self.output_layer(out)
        psi = psi.squeeze(dim=1)

        return psi

class CANN_simple(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor([10.0, 10.0]), requires_grad=True)
        self.b = nn.Parameter(torch.tensor([10.0, 10.0]), requires_grad=True)

    def forward(self, F):

        # clamp parameters without loop
        if True:
            parameters = torch.nn.utils.parameters_to_vector(self.parameters())
            parameters[parameters < max(parameters)*1e-3] = 0
            torch.nn.utils.vector_to_parameters(parameters, self.parameters())

        C = torch.matmul(F.transpose(-1,-2),F) # C = F^T * F
        I1 = torch.einsum('ijj -> i', C) # I1 = tr(C)
        J = torch.det(F)

        #out = (F**2 + self.b * F + self.c).sum(dim=(1, 2))
        out = self.a[0] * (I1 - 2) + self.a[1]*(J - 1)**2 + self.b[0] * (I1 - 2)**2 + self.b[1]*(J - 1)# + self.c[0] * (J-1) + self.c[1]*(J-1)**2
        return out

# ---------------------------------------------------------------------------
# ---------------------------- Initialization -------------------------------
# ---------------------------------------------------------------------------

# paths
dir_path = Path(__file__).parent.absolute()
results_path = Path(dir_path) / 'results'
log_path = Path(dir_path) / 'results' / 'log'

if log_path.exists():
    for file in os.listdir(log_path):
        os.remove(log_path / file)
else:
    log_path.mkdir(parents=True, exist_ok=True)

save_epochs = [0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700, 750, 800, 850, 900, 950, 1000]

# load data
u_list = np.load(dir_path / 'data/u.npy')
x = np.load(dir_path / 'data/x.npy')
beam_ty = np.load(dir_path / 'data/ty.npy')
beam_tx = np.load(dir_path / 'data/tx.npy')

u_list = torch.tensor(u_list, dtype=torch.float32, requires_grad=True)
x = torch.tensor(x, dtype=torch.float32, requires_grad=True)

grid_shape = (cf.Nx, cf.Ny)
dxdy = [cf.hx, cf.hy]

# setup domain for all different versions of boundary conditions
# only Neumann boundary conditions are changed
boundary_neumann_list = []
tractions = []
for i, (tx, ty) in enumerate(zip(beam_tx, beam_ty)):
    dom, boundary_neumann, boundary_dirichlet = des.setup_domain(traction_x=tx, traction_y=ty)
    boundary_neumann_list.append(boundary_neumann)
    tractions.append((tx, ty))
    print(f"loaded load state {i}: traction_x = {tx}, traction_y = {ty}")

dom = torch.from_numpy(dom).float().to(device)
dom.requires_grad_(True)
z = np.array([0])

num_datasets = len(boundary_neumann_list)

# ---------------------------- Hyperparameters ------------------------------

### most relevant ###
inner_optimizer = torchopt.MetaAdamW
inner_lr_main = 0.002 # 0.001
inner_convergence_crit_main = 0.005 #1e-5 # 1e-3
inner_conv_crit_outer_loss_exp = 0.9 #1.8 #1.4 # 1.2
inner_net_hl = 8
inner_comp_factor_start = 100
inner_lr_conv_factor_main = 33

outer_lr = 15.0
outer_weight_decay = 0.0
outer_optim_reset_period = 10000 # 100
outer_optim_reset_period_end = 10000 # 1000
outer_beta_1 = 0.9
outer_beta_2 = 0.8

### only relevant for starting phase ###
start_phase_epochs = 10

inner_convergence_crit_start = 1e-4 # 1e-4
inner_lr_start = 0.02 # 0.01

outer_loss_conv_crit = 8e-2 #1e-2

### probably not relevant ###
inner_max_epochs = 700
outer_loss_factor = 1e3

# ---------------------------------------------------------------------------
# overwriting hyperparameters for testing
outer_lr = 10
inner_max_epochs = 1000
inner_comp_factor_start = 2000
outer_beta_2 = 0.85
inner_lr_conv_factor_main = 30
# ---------------------------------------------------------------------------

# if True, the outer optimizer is only activated after all inner optimizers have converged
batched_outer_optimizer = True

# define outer_net
outer_net = CANN()
outer_optimizer = torch.optim.Adam(outer_net.parameters(), lr=outer_lr, betas=(outer_beta_1, outer_beta_2), fused=True)
outer_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(outer_optimizer, mode='min', factor=0.8,
                                                             patience=20, threshold=0.01, threshold_mode='rel',
                                                             cooldown=20, min_lr=0.2, eps=1e-08, verbose=True)

# initialize loop variables
inner_losses = [[] for _ in range(num_datasets)]
outer_losses = [[] for _ in range(num_datasets)]
param1 = [[] for _ in range(num_datasets)]
param2 = [[] for _ in range(num_datasets)]
outer_optimizer_active = [False]*num_datasets
batched_outer_optimizer_active = False
accumulated_outer_losses = []

# initialize list of dem models and states
dem_model = [None] * num_datasets
dem_state = [None] * num_datasets
u_dem = [None] * num_datasets

# define colors for plots; same as standard matplotlib colors
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

# initialize plots

plot_legend = [f'$t_{i} = [{tractions[i][0], tractions[i][1]}]$' for i in range(num_datasets)]
beam_plots = [BeamPlotter(x, grid_shape[0], grid_shape[1], labels=["Given x", "Predicted x"]) for _ in range(num_datasets)]

if batched_outer_optimizer:
    outer_loss_plot = LossPlot(num_lines=1, title='CANN Loss Over Epoch', xlabel='CANN Epoch', ylabel='CANN Loss', log_y=True)
else:
    outer_loss_plot = LossPlot(num_lines=num_datasets, title='CANN Loss Over Epoch', xlabel='CANN Epoch', ylabel='CANN Loss', log_y=True, legend=plot_legend)

inner_loss_plot = LossPlot(num_lines=num_datasets, title='DEM Loss Over Epoch', xlabel='Cumulative DEM Epoch', ylabel='DEM Loss', legend=plot_legend)
param_loss_plot = ParamLossPlot(num_datasets=num_datasets, title='Parameter Loss Surface')

inner_iter = 0
outer_iter = 0
# ---------------------------------------------------------------------------
# ---------------------------- Training Loop --------------------------------
# ---------------------------------------------------------------------------

#try:
if True:
    for outer_epoch in range(2000):

        # (re)initialize outer optimizer to ensure fast convergence
        if outer_epoch > 0:
            if outer_epoch % outer_optim_reset_period == 0 and accumulated_outer_losses[-1] > 5:
                if batched_outer_optimizer:
                    outer_optimizer = torch.optim.Adam(outer_net.parameters(), lr=outer_lr, betas=(outer_beta_1, outer_beta_2), fused=True)
                else:
                    for i in range(num_datasets):
                        outer_optimizer = torch.optim.Adam(outer_net.parameters(), lr=outer_lr, betas=(outer_beta_1, outer_beta_2), fused=True)

        accumulated_outer_loss = 0
        dataset_order = np.random.permutation(num_datasets)
        for dataset_idx in dataset_order:
            u = u_list[dataset_idx]

            if outer_epoch < start_phase_epochs:
                inner_lr = inner_lr_start
                inner_conv_crit = inner_convergence_crit_start
                inner_comp_factor = inner_comp_factor_start
            else:
                inner_lr = inner_lr_main * (outer_losses[dataset_idx][-1]/outer_loss_factor)**0.2
                inner_conv_crit = inner_lr * (1/inner_lr_conv_factor_main) * (outer_losses[dataset_idx][-1]/outer_loss_factor)**0.25
                inner_comp_factor = inner_comp_factor_start * (outer_losses[dataset_idx][-1]/outer_loss_factor)**0.5

            # (re)initialize inner_net
            if outer_iter == 0 or inner_iter > 20:
                inner_iter = 0
                
                for i in range(len(dem_model)):
                    dem_model[i] = DeepEnergyMethod([2, inner_net_hl, 2], 'trapezoidal', outer_net, 2, inner_lr,
                                            device, model_type='CANN', optimizer=inner_optimizer)

                    if dem_state[i] is not None:
                        dem_model[i].model.load_state_dict(dem_state[i]['model_state_dict'])

                        if all(outer_optimizer_active):
                            noise_level = 0.005
                            for param in dem_model[i].model.parameters():
                                if param.requires_grad:
                                    param.data += noise_level * torch.randn_like(param)
    
            # -----------------------------------------------------------------
            # ----------------------- DEM training loop -----------------------
            # -----------------------------------------------------------------
            dem_state[dataset_idx], dem_loss = dem_model[dataset_idx].train_model(grid_shape, dxdy, dom,
                                                        boundary_neumann_list[dataset_idx], boundary_dirichlet, inner_max_epochs,
                                                        convergence_criterion=inner_conv_crit, compression_factor=inner_comp_factor)[2:4]
            # -----------------------------------------------------------------	
            
            inner_losses[dataset_idx].extend(dem_loss)
            inner_loss_plot.update(inner_losses, epochs=100000)
            inner_loss_plot.add_vertical_line(active_line=dataset_idx, line_color=colors[dataset_idx])

            inner_iter += len(dem_loss)
            
            u_dem[dataset_idx] = dem_model[dataset_idx].getU(x)
            
            outer_loss = (u_dem[dataset_idx] - u).pow(2).sum()*outer_loss_factor/len(u)
            accumulated_outer_loss += outer_loss

            outer_losses[dataset_idx].append(outer_loss.item())

            if isinstance(outer_net, CANN_simple):
                weights = outer_net.a.data                        
                param1[dataset_idx].append(weights[0].item())
                param2[dataset_idx].append(weights[1].item())   

            elif isinstance(outer_net, CANN):
                weights = outer_net.output_layer.weight.data
                param1[dataset_idx].append(weights[0][0].item())
                param2[dataset_idx].append(weights[0][3].item())  
            
            current_outer_lr = outer_optimizer.param_groups[0]['lr']
            
            if not batched_outer_optimizer:
                param_loss_plot.update(param1, param2, outer_losses, fraction=0.8)
                outer_loss_plot.update(outer_losses, epochs=2000)

                print(f'outer_epch = {outer_epoch} outer_loss = {outer_loss:.2f},'
                    f'inner_conv_crit = {inner_conv_crit:.2e}, inner_lr = {inner_lr:.2e},'
                    f'compression_factor = {inner_comp_factor:.2f}, outer_lr = {current_outer_lr:.2e}')
            
                # detect initial outer loss convergence to activate outer_optimizer
                if outer_epoch >= 10 and not outer_optimizer_active[dataset_idx]:
                    outer_losses_end = outer_losses[dataset_idx][-10:]
                    if max(outer_losses_end) - min(outer_losses_end) < outer_loss_conv_crit*outer_losses[dataset_idx][-1]:
                        print('-------- outer loss convergence detected --------')
                        outer_loss_plot.add_vertical_line(active_line=dataset_idx, line_color='blue')
                        outer_optimizer_active[dataset_idx] = True

                # outer_optimizer step
                if all(outer_optimizer_active) and len(dem_loss) < inner_max_epochs:
                    outer_scheduler.step(outer_loss)
                    outer_optimizer.zero_grad()
                    outer_loss.backward(retain_graph=True)
                    outer_optimizer.step()
                    beam_plots[dataset_idx].update([u, u_dem[dataset_idx]], new_title=f'Beam - Epoch {outer_epoch}')
                    if outer_epoch in save_epochs:
                        beam_plots[dataset_idx].save(log_path, filename=f'beam_plot_{outer_epoch}_dattaset_{dataset_idx}')

                outer_iter += 1

        accumulated_outer_losses.append(accumulated_outer_loss.item())
        if batched_outer_optimizer:
            param_loss_plot.update(param1, param2, [accumulated_outer_losses], fraction=0.8)
            outer_loss_plot.update([accumulated_outer_losses], epochs=2000)

            print(f'outer_epch = {outer_epoch} outer_loss = {accumulated_outer_loss:.2f},'
                  f'inner_conv_crit = {inner_conv_crit:.2e}, inner_lr = {inner_lr:.2e},'
                  f'compression_factor = {inner_comp_factor:.2f}, outer_lr = {current_outer_lr:.2e}')
            
            # detect initial outer loss convergence to activate outer_optimizer
            if outer_epoch >= 10 and not batched_outer_optimizer_active:
                outer_losses_end = accumulated_outer_losses[-10:]
                if max(outer_losses_end) - min(outer_losses_end) < outer_loss_conv_crit*accumulated_outer_losses[-1]:
                    print('-------- outer loss convergence detected --------')
                    outer_loss_plot.add_vertical_line(active_line=0, line_color='blue')
                    batched_outer_optimizer_active = True

            # outer_optimizer step
            if batched_outer_optimizer_active and len(dem_loss) < inner_max_epochs:
                outer_scheduler.step(accumulated_outer_loss)
                outer_optimizer.zero_grad()
                accumulated_outer_loss.backward(retain_graph=True)
                outer_optimizer.step()
                beam_plots[dataset_idx].update([u, u_dem[dataset_idx]], new_title=f'Beam - Epoch {outer_epoch}')
                if outer_epoch in save_epochs:
                    beam_plots[dataset_idx].save(log_path, filename=f'beam_plot_{outer_epoch}_dattaset_{dataset_idx}.svg')

#except KeyboardInterrupt:
#    print("Training interrupted by the user.")
#except Exception as e:
#    # catch any error that occurs during training and print it with the current epoch and line number
#    exc_type, exc_obj, exc_tb = sys.exc_info()
#    print(f"Error at line {exc_tb.tb_lineno}: {e}")

beam_plots[dataset_idx].update([u, u_dem[dataset_idx]], new_title=f'Original and Deformed Beam - Epoch {outer_epoch}')
beam_plots[dataset_idx].save(log_path)

# rename the figure folder to include beginning loss and end loss and the number of epochs and the number of datasets
log_path_new = os.path.join(results_path, f'plots_{outer_losses[0][0]:.2f}_{outer_losses[0][-1]:.2f}_{outer_epoch+1}_{num_datasets}')
os.rename(log_path, log_path_new)

# save data
np.save(log_path_new + '/u.npy', u_list.detach().cpu().numpy())
np.save(log_path_new + '/x.npy', x.detach().cpu().numpy())
np.save(log_path_new + '/ty.npy', beam_ty)
outer_loss_plot.save(log_path_new, filename='loss_log')

#np.save(log_path_new + '/outer_losses.npy', outer_losses)
#np.save(log_path_new + '/inner_losses.npy', inner_losses)

# print coefficients
#print(weights)

# Print all parameters
for name, param in outer_net.named_parameters():
    print(name, param.data)
    # save parameters to txt file; create file if it does not exist
    with open(log_path_new + '/parameters.txt', 'a') as file:
        file.write(f'{name} {param.data}\n')

try:
    print(outer_net.b.data)
except:
    pass

plt.show(block=True)

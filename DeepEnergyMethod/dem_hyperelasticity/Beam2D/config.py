
# ------------------------------ network settings ---------------------------------------------------
iteration = 20
D_in = 2

### Jonathan ###
H = 30 # 8
###############

D_out = 2
### Jonathan ###
#lr = 0.1
lr = 0.01
################
# ------------------------------ material parameter -------------------------------------------------
model_energy = 'NeoHookean2D'
E = 1000

### Jonathan ###
#nu = 0.3
nu = 0.49
################

param_c1 = 630
param_c2 = -1.2
param_c = 100
# ----------------------------- define structural parameters ---------------------------------------
Length = 4.0
Height = 1.0
Depth = 1.0
known_left_ux = 0
known_left_uy = 0
bc_left_penalty = 1.0

known_right_tx = 0

### Jonathan ###
known_right_ty = -5.0
#known_right_ty = -0.5
################

bc_right_penalty = 1.0
# ------------------------------ define domain and collocation points -------------------------------

### Jonathan ###
Nx = 20 # 200 # 120  # 120
Ny = 5 # 50 # 30  # 60
################

x_min, y_min = (0.0, 0.0)
hx = Length / (Nx - 1)
hy = Height / (Ny - 1)
shape = [Nx, Ny]
dxdy = [hx, hy]

# ------------------------------ data testing -------------------------------------------------------

### Jonathan ###
num_test_x = Nx #200
num_test_y = Ny #50
################

import torch

def compute_F(x, u, grid_shape):
    # detect which of the coordinates of [x, y] changes first in the array x
    if x[0,0] != x[1,0]:
        # the first coordinate, x, changes first
        x = x.reshape(grid_shape[::-1] + (2,)).transpose(0, 1)
        u = u.reshape(grid_shape[::-1] + (2,)).transpose(0, 1)

    elif x[0,1] != x[1,1]:
        # the second coordinate, y, changes first
        x = x.reshape(grid_shape + (2,))
        u = u.reshape(grid_shape + (2,))

    else:
        raise ValueError('x does not have the correct shape')
    
    # Get the x and y coordinates for spacing
    x_coords = x[:,0,0]
    y_coords = x[0,:,1]

    # Compute the gradients of u with respect to x using torch
    du_dx, du_dy = torch.gradient(u[..., 0], spacing=(x_coords, y_coords))
    dv_dx, dv_dy = torch.gradient(u[..., 1], spacing=(x_coords, y_coords))

    # Compute the components of the deformation gradient tensor
    F_xx = 1 + du_dx
    F_xy = du_dy
    F_yx = dv_dx
    F_yy = 1 + dv_dy

    # Stack the components to get the deformation gradient tensor in shape (Nx, Ny, 2, 2) using torch
    F = torch.stack((torch.stack((F_xx, F_xy), dim=-1), torch.stack((F_yx, F_yy), dim=-1)), dim=-1)

    # Reshape the deformation gradient tensor to the original shape of x and u (10000, 2, 2)
    F = F.reshape((-1, 2, 2))

    return F
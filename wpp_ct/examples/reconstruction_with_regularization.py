

import torch 
import odl 
from odl.contrib.torch import OperatorModule

from tqdm import tqdm
import numpy 
import matplotlib.pyplot as plt 


def tv_loss(x):
    """
    Anisotropic TV loss similar to the one in [1]_.

    Parameters
    ----------
    x : :class:`torch.Tensor`
        Tensor of which to compute the anisotropic TV w.r.t. its last two axes.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Total_variation_denoising
    """
    dh = torch.abs(x[..., :, 1:] - x[..., :, :-1])
    dw = torch.abs(x[..., 1:, :] - x[..., :-1, :])
    return torch.sum(dh[..., :-1, :] + dw[..., :, :-1])


reco_space = odl.uniform_discr(
            min_pt=[-256, -256], max_pt=[256, 256], shape=[512, 512])

phantom = odl.phantom.shepp_logan(reco_space, modified=True)

geometry = odl.tomo.parallel_beam_geometry(reco_space, 350)
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')

sinogram = ray_trafo(phantom)

fig, (ax1, ax2) = plt.subplots(1,2)

ax1.imshow(phantom)
ax1.set_title("Shepp Logan Phantom")

ax2.imshow(sinogram)
ax2.set_title("Sinogram")
ax2.set_xlabel("detector pixels")
ax2.set_ylabel("angles")
ax2.set_aspect("auto")

plt.show()

forward_operator = OperatorModule(ray_trafo)
xt = torch.nn.Parameter(torch.zeros(phantom.shape, device="cuda").unsqueeze(0).unsqueeze(0), requires_grad=True)
optimizer = torch.optim.SGD([xt], lr=1/(2*odl.power_method_opnorm(ray_trafo)**2))

sinogram_torch = torch.from_numpy(sinogram.asarray()).unsqueeze(0).unsqueeze(0)
sinogram_torch = sinogram_torch.to("cuda")
alpha = 1.

print(xt.device, sinogram_torch.device)
for i in tqdm(range(400)):
    optimizer.zero_grad() 

    mse_term = torch.nn.functional.mse_loss(forward_operator(xt), sinogram_torch)
    tv_term = tv_loss(xt)
    #print(mse_term.item(), tv_term.item())
    loss = mse_term + alpha*tv_term
    loss.backward()

    optimizer.step()


fig, (ax1, ax2) = plt.subplots(1,2)

ax1.imshow(phantom)

ax2.imshow(xt.detach().cpu().numpy()[0,0,:,:])

plt.show()
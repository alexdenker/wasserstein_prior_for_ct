

from torch.utils.data import DataLoader
from dival import get_standard_dataset

import matplotlib.pyplot as plt 

impl = 'astra_cuda' # 'astra_cpu'
sorted_by_patient = True # False 

# create a dival dataset 
dataset = get_standard_dataset('lodopab', impl=impl,
                        sorted_by_patient=sorted_by_patient)
ray_trafo = dataset.get_ray_trafo(impl=impl) # OperatorModule(ray_trafo) -> nn.Module

# use inbuild method .create_torch_dataset
lodopab_train = dataset.create_torch_dataset(part='train',
                                    reshape=((1,) + dataset.space[0].shape,
                                    (1,) + dataset.space[1].shape))
            
lodopab_val = dataset.create_torch_dataset(part='validation',
                                    reshape=((1,) + dataset.space[0].shape,
                                        (1,) + dataset.space[1].shape))

lodopab_test = dataset.create_torch_dataset(part='test',
                                    reshape=((1,) + dataset.space[0].shape,
                                        (1,) + dataset.space[1].shape))

# create data loader 
batch_size = 16
num_data_loader_workers = 16
dl = DataLoader(lodopab_train, batch_size=batch_size,
                          num_workers=num_data_loader_workers,
                          shuffle=True, pin_memory=True)

for idx, (y,x) in enumerate(dl): # output is ordered as (sinogram, ground truth image)
    print(x.shape, y.shape)

    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(x[0,0,:,:], cmap="gray")
    ax2.imshow(y[0,0,:,:])

    plt.show()
    break
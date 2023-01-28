# Pytorch tutorial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# 1. Tensors
# 1.1. Tensor creation
# 1.1.1. Directly from data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# 1.1.2. From a NumPy array
import numpy as np
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# 1.1.3. From another tensor
x_ones = torch.ones_like(x_data)  # retains the properties of x_data
print(f"Ones Tensor: {x_ones}")

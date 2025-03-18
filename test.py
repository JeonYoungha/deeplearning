import torch
import numpy as np

device = "mps" if torch.mps.is_available() else "cpu"


t1 = torch.tensor([1,2,3,4], dtype=torch.float16, device=device,requires_grad=False)
type(t1)
print(t1)

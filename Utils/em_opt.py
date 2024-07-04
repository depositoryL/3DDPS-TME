import torch
import numpy as np


def traffic_em(flows_loads, links_loads, rm, n):
    b = flows_loads.shape[0]
    flows_loads_final = flows_loads.clone()

    for i in range(b):
        flows_loads_i = em_step(flows_loads[i], links_loads[i], rm, n)
        flows_loads_final[i] = flows_loads_i

    return flows_loads_final


def em_step(x, y, rm, n):
    x = replace_neg(x).to(x.device)
    rm = rm.to(x.device)
    minloss = 100
    x_final = x
    for i in range(n):
        temp_y = x @ rm
        temp_y = replace_neg(temp_y)
        a = torch.div(x, rm.sum(dim=1))
        b = torch.div(rm, temp_y)
        c = y @ torch.transpose(b, 0, 1)
        x = torch.mul(a, c)
        temploss = torch.abs(x @ rm - y)
        temploss = temploss.sum(dim=0)

        if temploss < minloss:
            minloss = temploss
            x_final = x

    return x_final
    

def replace_neg(x):
    x1 = x.detach().cpu().numpy()
    x_0_idx = np.argwhere(x1 < 0)
    x1[x_0_idx] = 1
    xmin = np.min(x1)
    x[x_0_idx] = torch.tensor(xmin)
    return x
#!/usr/bin/env python3
import torch
import torch as th
import numpy as np
policy = torch.jit.load('../pre_train/g1/policy_eetrack.pt')
obs=np.load('/tmp/eet5/obs002.npy')
print('obs', obs.shape)
act=policy(torch.from_numpy(obs))
act_sim=np.load('/tmp/eet5/act002.npy')
act_rec=act.detach().cpu().numpy()
delta= (act_sim - act_rec)
print(np.abs(delta).max())

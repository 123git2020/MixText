import subprocess
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
kl_loss = nn.KLDivLoss(reduction="batchmean")

x=torch.tensor([[0.1,0.2,0.7],[0.1,0.85,0.05]])
y=torch.tensor([[0.1,0.1,0.8],[0.1,0.7,0.2]])

# input should be a distribution in the log space
input = F.log_softmax(torch.randn(3, 5, requires_grad=True), dim=1)
# Sample a batch of distributions. Usually this would come from the dataset
target = F.softmax(torch.rand(3, 5), dim=1)
output = kl_loss(x.log(), y)

v=y*(-x.log()+y.log())

def lian():
    result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
    output = result.stdout
    for line in output.splitlines():
        if '=' in line:
            var, value = line.split('=', 1)
            os.environ[var] = value
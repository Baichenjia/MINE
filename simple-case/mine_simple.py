# MINE: Mutual Information Neural Estimation
# https://github.com/MasanoriYamada/Mine_pytorch/blob/master/mine.ipynb

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# data
var = 0.2
def func(x):
    return x

def gen_x(data_size):
    return np.sign(np.random.normal(loc=0., scale=1., size=[data_size,1]))

def gen_y(x, data_size):
    return func(x) + np.random.normal(loc=0., scale=np.sqrt(var), size=[data_size,1])


x = gen_x(data_size=20000)
y = gen_y(x, data_size=20000)       # y服从均值为x, 方差为var的分布

# Calculate mutual information using traditional method
p_y_x = np.exp(-(y-x)**2/(2*var))         # 考虑 x,y 相互关联时
p_y_x_minus = np.exp(-(y+1)**2/(2*var))   # 当 x=-1 时
p_y_x_plus = np.exp(-(y-1)**2/(2*var))    # 当 x= 1 时
mi = np.average(np.log(p_y_x/(0.5*p_y_x_minus+0.5*p_y_x_plus)))   # KL[p(x,y) || p(x)*p(y)]
print("Mutual Information by Traditional Method:", mi)

# MINE
class Net(nn.Module):
    def __init__(self):
    	# 求 T(x,y), 输入 x,y 输出 T
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(1, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x, y):
        h1 = F.relu(self.fc1(x)+self.fc2(y))
        h2 = self.fc3(h1)
        return h2


model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_total = []

for epoch in tqdm(range(500)):
    x_sample = torch.as_tensor(gen_x(data_size=20000), dtype=torch.float32)              # x
    y_sample = torch.as_tensor(gen_y(x_sample, data_size=20000), dtype=torch.float32)    # y 与 x 是有关的
    y_shuffle = torch.as_tensor(np.random.permutation(y_sample), dtype=torch.float32)    # y 与 x 是独立的
    
    # use samples as the trainable parameter
    x_sample = torch.nn.Parameter(x_sample)
    y_sample = torch.nn.Parameter(y_sample)
    y_shuffle = torch.nn.Parameter(y_shuffle)

	# MINE
    pred_xy = model(x_sample, y_sample)        # T(X,Y) where Y from Joint
    pred_x_y = model(x_sample, y_shuffle)      # T(X,Y) where Y from Marginal
    ret = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))  # I(X,Y)

    # train
    model.zero_grad()
    loss = -ret                                # maximize the I(X,Y)
    loss.backward()                            # compute the gradient
    optimizer.step()                           # step the gradient
    loss_total.append(-loss.item())            # record the mutual information
    # print("epoch:", epoch, ", loss:", loss.item())
    
plt.plot(np.arange(len(loss_total)), loss_total, label="MINE")                  # MINE
plt.plot(np.arange(len(loss_total)), [mi]*len(loss_total), label="Traditional") # traditional 
plt.xlabel("training epoch")
plt.ylabel("mutual information")
plt.legend()
plt.savefig("mine_simple.jpg", dpi=300)
plt.show()


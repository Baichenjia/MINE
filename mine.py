# 生成二元高斯数据, 求不同维度之间数据的互信息

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

#########################################
### 1. Generate Gaussian ################
#########################################

def Generate_Gaussian(plot=True):
    # 生成独立高斯, 不同维度之间独立. shape=(300, 2)
    x = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=300)
    # 生成多元高斯, 不同维度之间相互依赖. shape=(300, 2)
    y = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0.8], [0.8, 1]], size=300)
    
    if plot:
        plt.scatter(x[:, 0], x[:, 1], label="Diagonal Normal")
        plt.scatter(y[:, 0], y[:, 1], label="Multivariate Normal")
        plt.legend()
        plt.savefig("mine_data.jpg", dpi=300)

    return x, y 


###################################################
### 2. Compute Mutual Information Using MINE ######
###################################################

class Mine(nn.Module):
    def __init__(self, input_size=2, hidden_size=100):
        super(Mine, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.fc1.bias, val=0)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.fc2.bias, val=0)
        nn.init.normal_(self.fc3.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.fc3.bias, val=0)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        y = self.fc3(x)
        return y

def learn_mine(batch, mine_net, mine_net_optim, avg_et=1.0, unbiased_loss=True):
    batch = [torch.from_numpy(batch[0]), torch.from_numpy(batch[1])]
    joint, marginal = batch[0], batch[1]
    # 数据也被当做参数进行运行，这里可以考虑不用，只训练网络.
    # joint = torch.nn.Parameter(batch[0])
    # marginal = torch.nn.Parameter(batch[1])

    # calculate mutual information using MINE
    T_joint = mine_net(joint)                          # (None, 1)
    T_marginal = mine_net(marginal)                    # (None, 1)
    T_marginal_exp = torch.exp(T_marginal)
    
    # calculate loss
    if unbiased_loss:                          # unbiased gradient
        avg_et = 0.99 * avg_et + 0.01 * torch.mean(T_marginal_exp)
        # mine_estimate_unbiased = torch.mean(T_joint) - (1/avg_et).detach() * torch.mean(T_marginal_exp)
        mine_estimate = torch.mean(T_joint) - (torch.mean(T_marginal_exp)/avg_et).detach() * torch.log(torch.mean(T_marginal_exp))
        loss = -1. * mine_estimate
    else:                                      # biased gradient
        mine_estimate = torch.mean(T_joint) - torch.log(torch.mean(T_marginal_exp))
        loss = -1. * mine_estimate

    # calculate gradient and train
    mine_net.zero_grad()
    loss.backward()
    mine_net_optim.step()
    return mine_estimate, avg_et

###################################################
### 3. Construct Joint sample and Marginal sample #
###################################################

def sample_batch(data, batch_size=100, sample_mode='joint'):
    # 对于一组数据的两个维度，构造 joint sample 和 marginal sample. 
    # 如果这组数据中的两个维度本来就互相独立，那么 joint sample 和 marginal sample 的分布是一致的.
    assert sample_mode in ['joint', 'marginal']
    index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
    batch = data[index]           # (batch_size, 2)

    if sample_mode is 'joint':
        return batch.astype(np.float32)
    elif sample_mode is 'marginal':
        # 独立的采样另外一批数据，提取第二维。使用之前采样的第一维。二者连接起来，这样两维之间没有依赖关系
        new_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        new_batch = data[new_index]                                 # (batch_size, 2)
        # 将 batch 和 new_batch 组合起来，这样二者之间互不关联
        return np.stack([batch[:, 0], new_batch[:, 1]], axis=1).astype(np.float32)  # (batch_size, 2)

def test_sample_batch():
    x, y = Generate_Gaussian(plot=False)

    # 输入的数据是 y, 该数据的两个维度之间本来是相关的
    # 1. 采样 joint 数据, 即直接采样 y
    joint_data = sample_batch(y, batch_size=100, sample_mode='joint')
    plt.scatter(joint_data[:,0], joint_data[:,1], color='red', label='Joint sample', zorder=4)
    
    # 2. 采样 marginal 数据, 即独立的采样两个维度
    marginal_data = sample_batch(y,batch_size=100,sample_mode='marginal')
    plt.scatter(marginal_data[:,0], marginal_data[:,1], label='Marginal sample')
    plt.legend()
    plt.title("Sample From Joint Data")
    plt.savefig("Joint_Margine_Sample.jpg", dpi=300)


#####################################################
### 4. Train the function with data #################
#####################################################

def train(data, mine_net, mine_net_optim, batch_size=100, iter_num=5000, log_freq=int(1e+3)):
    result = []
    avg_et = 1.
    for i in range(iter_num):
        # 产生联合分布的数据和边缘分布的数据
        joint_data = sample_batch(data, batch_size=batch_size, sample_mode='joint')
        marginal_data = sample_batch(data, batch_size=batch_size, sample_mode='marginal')
        # 训练模型，返回目前的 mutual-information
        mi_lb, avg_et = learn_mine([joint_data, marginal_data], mine_net, mine_net_optim, avg_et)
        result.append(mi_lb.detach().numpy())
        if (i+1) % (log_freq) == 0:
            print("iter:", i, ", MI:", mi_lb.detach().numpy())
    return result

def ma(a, window_size=100):
    # 对整个结果进行滑动窗口平均
    return [np.mean(a[i: i+window_size]) for i in range(0, len(a)-window_size)]


#########################################################
### 5. Run Construct Joint sample and Marginal sample ###
#########################################################

def train_independent_data():
    # data
    x, _ = Generate_Gaussian(plot=False)

    # train
    mine_net_indep = Mine()
    mine_net_optim_indep = torch.optim.Adam(mine_net_indep.parameters(), lr=1e-3)
    result_indep = train(x, mine_net_indep, mine_net_optim_indep)

    # plot
    result_indep_ma = ma(result_indep)
    print("Mutual Information Indep:", result_indep_ma[-1])
    plt.clf()
    plt.plot(range(len(result_indep_ma)), result_indep_ma)
    plt.savefig("Independent_Mutual_Info.jpg", dpi=300)


def train_correlation_data():
    # data
    _, y = Generate_Gaussian(plot=False)

    # train
    mine_net_corr = Mine()
    mine_net_optim_corr = torch.optim.Adam(mine_net_corr.parameters(), lr=1e-3)
    result_corr = train(y, mine_net_corr, mine_net_optim_corr)

    # plot
    result_corr_ma = ma(result_corr)
    print("Mutual Information Corr:", result_corr_ma[-1])
    plt.clf()
    plt.plot(range(len(result_corr_ma)), result_corr_ma)
    plt.savefig("Correlation_Mutual_Info.jpg", dpi=300)


def train_correlation_various():
    plt.clf()
    # correction
    correlations = np.linspace(-0.9, 0.9, 19)
    
    # generate data and train
    final_result = []
    for rho in correlations:
        rho_data = np.random.multivariate_normal(mean=[0,0], cov=[[1,rho],[rho,1]], size=300)
        mine_net = Mine()
        mine_net_optim = torch.optim.Adam(mine_net.parameters(), lr=1e-3)
        result = train(rho_data, mine_net, mine_net_optim)
        result_ma = ma(result)
        plt.plot(range(len(result_ma)),result_ma, label="corr:"+str(round(rho, 1)))  # curve during training

        # record final result
        final_result.append(result_ma[-1])
    plt.legend(ncol=4, loc='upper center')
    plt.ylim([-0.1, 1.5])
    plt.grid()
    plt.savefig("Mutual-Information-of-Different-Corrections-during-Training.jpg", dpi=300)

    plt.clf()
    plt.plot(correlations, final_result)
    plt.savefig("Mutual-Information-of-Different-Corrections-Final.jpg", dpi=300)


#########################################################
### 6. Using Specific correlation  ######################
#########################################################

def train_specific_correlation():
    plt.clf()
    # 以上只测试了高斯情况下的 mutual-information, 现在测试几个特定的函数
    x = np.random.uniform(low=-1.0, high=1.0, size=3000)
    f1 = x 
    f2 = 2 * x
    f3 = np.sin(x)
    f4 = x ** 3 
    eps = np.random.normal(size=3000)
    
    sigmas = np.linspace(0.0, 0.9, 4)      # 代表噪声
    fs = [f1, f2, f3, f4]

    final_result = []
    for sigma in sigmas:
        for fi, f in enumerate(fs):
            data = np.concatenate([x.reshape(-1, 1), (f+sigma*eps).reshape(-1, 1)], axis=1)
            mine_net = Mine()
            mine_net_optim = torch.optim.Adam(mine_net.parameters(), lr=1e-3)
            result = train(data, mine_net, mine_net_optim, iter_num=1000)
            result_ma = ma(result)
            final_result.append(result_ma[-1])
            plt.plot(range(len(result_ma)), result_ma, label="f"+str(fi+1)+"-sigma-"+str(round(sigma, 1)))

    plt.legend(ncol=4, loc='upper center')
    plt.ylim([-0.1, 10])
    plt.grid()
    plt.savefig("Mutual-Information-of-Specific-Function.jpg", dpi=300)


if __name__ == '__main__':
    # train_independent_data()
    # train_correlation_data()
    train_correlation_various()
    # train_specific_correlation()

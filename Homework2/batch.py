from utils import DGraphFin
from utils.utils import prepare_folder
from utils.evaluator import Evaluator
from utils.model import MLP,SAGE,GCN
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.transforms as T
import numpy as np
from torch_geometric.data import Data
import os



#设置gpu设备
device = 0
device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
print(device)



path='./datasets/632d74d4e2843a53167ee9a1-momodel/' #数据保存路径
save_dir='./results/' #模型保存路径
dataset_name='DGraph'
dataset = DGraphFin(root=path, name=dataset_name,transform=T.ToSparseTensor())



mlp_parameters = {
    'lr': 1e-2
    , 'num_layers': 2
    , 'hidden_channels': 32
    , 'dropout': 0.01
    , 'batchnorm': False
    , 'weight_decay': 5e-7
    , 'need_edge_index':False
                  }
sage_parameters = {
    'lr': 1e-2
    , 'num_layers': 2
    , 'hidden_channels': 16
    , 'dropout': 0.1
    , 'batchnorm': False
    , 'weight_decay': 5e-7
    , 'need_edge_index':True
                  }

gcn_parameters = {
    'lr': 1e-2
    , 'num_layers': 1
    , 'hidden_channels': 16
    , 'dropout': 0.1
    , 'batchnorm': False
    , 'weight_decay': 5e-7
    , 'need_edge_index':True
                  }

model_list = ['mlp','sage','gcn']

model_name = model_list[2]
parameters_name = model_name+'_parameters'


nlabels = dataset.num_classes
if dataset_name in ['DGraph']:
    nlabels = 2    #本实验中仅需预测类0和类1

data = dataset[0]
data.adj_t = data.adj_t.to_symmetric() #将有向图转化为无向图



if dataset_name in ['DGraph']:
    x = data.x
    x = (x - x.mean(0)) / x.std(0)
    data.x = x
if data.y.dim() == 2:
    data.y = data.y.squeeze(1)

split_idx = {'train': data.train_mask, 'valid': data.valid_mask, 'test': data.test_mask}  #划分训练集，验证集

# loader = DataLoader(data_list, batch_size=2, follow_batch=['x_s', 'x_t'])
train_idx = split_idx['train']
result_dir = prepare_folder(dataset_name,model_name)

print(data)
print(data.edge_index)
print(data.x.shape)  #feature
print(data.y.shape)  #label



epochs = 200
log_steps =10 # log记录周期
para_dict = globals()[parameters_name]
model_para = globals()[parameters_name].copy()
need_edge_index = model_para['need_edge_index']
model_dict = {
    'mlp':MLP,
    'sage':SAGE,
    'gcn':GCN
}



model_para.pop('lr')
model_para.pop('weight_decay')
model_para.pop('need_edge_index')
model = model_dict[model_name](in_channels=data.x.size(-1), out_channels=nlabels, **model_para).to(device)
print(f'Model {model_name} initialized')


eval_metric = 'auc'  #使用AUC衡量指标
evaluator = Evaluator(eval_metric)

def train(model, data, train_idx, optimizer):
     # data.y is labels of shape (N, ) 
    model.train()

    optimizer.zero_grad()
    input_x = data.x[train_idx].to(device)
    input_y = data.y[train_idx].to(device)
    if need_edge_index:
        input_adj_t = data.adj_t[train_idx].to(device)
        out = model(input_x,input_adj_t)
        del input_adj_t
    else:
        out = model(input_x)
    loss = F.nll_loss(out, input_y)
    loss.backward()
    optimizer.step()
    del input_x,input_y
    return loss.item()

def test(model, data, split_idx, evaluator):
    # data.y is labels of shape (N, )
    with torch.no_grad():
        model.eval()

        losses, eval_results = dict(), dict()
        for key in ['train', 'valid']:
            node_id = split_idx[key]
            input_x = data.x[node_id].to(device)
            input_y = data.y[node_id].to(device)
            if need_edge_index:
                input_adj_t = data.adj_t[node_id].to(device)
                out = model(input_x,input_adj_t)
                del input_adj_t
            else:
                out = model(input_x)
            y_pred = out.exp()  # (N,num_classes)
            losses[key] = F.nll_loss(out, input_y).item()
            eval_results[key] = evaluator.eval(data.y[node_id], y_pred)[eval_metric]
            del input_x, input_y

    return eval_results, losses, y_pred

print(sum(p.numel() for p in model.parameters()))  #模型总参数量

# model.reset_parameters()
optimizer = torch.optim.Adam(model.parameters(), lr=para_dict['lr'], weight_decay=para_dict['weight_decay'])
best_valid = 0
min_valid_loss = 1e8

for epoch in range(1,epochs + 1):
    loss = train(model, data, train_idx, optimizer)
    eval_results, losses, out = test(model, data, split_idx, evaluator)
    train_eval, valid_eval = eval_results['train'], eval_results['valid']
    train_loss, valid_loss = losses['train'], losses['valid']

    if valid_loss < min_valid_loss:
        min_valid_loss = valid_loss
        torch.save(model.state_dict(), save_dir+'/model.pt',_use_new_zipfile_serialization=False) #将表现最好的模型保存

    if epoch % log_steps == 0:
        print(f'Epoch: {epoch:02d}, '
              f'Loss: {loss:.4f}, '
              f'Train: {100 * train_eval:.3f}, ' # 我们将AUC值乘上100，使其在0-100的区间内
              f'Valid: {100 * valid_eval:.3f} ')

model.load_state_dict(torch.load(save_dir+'/model.pt',map_location='cpu')) #载入验证集上表现最好的模型
def predict(data,node_id):
    """
    加载模型和模型预测
    :param node_id: int, 需要进行预测节点的下标
    :return: tensor, 类0以及类1的概率, torch.size[1,2]
    """
    # -------------------------- 实现模型预测部分的代码 ---------------------------
    with torch.no_grad():
        model.eval()
        input_x = data.x[node_id].to(device)
        if need_edge_index:
            input_adj_t = data.adj_t[node_id].to(device)
            out = model(input_x,input_adj_t)
        else:
            out = model(input_x)
        y_pred = out.exp()  # (N,num_classes)
        
    return y_pred

def save_test_result(data,node_id):
    with torch.no_grad():
        model.eval()
        out = model(data.x[node_id])
        y_pred = out.exp()  # (N,num_classes)
    return y_pred

test_idx = split_idx['test']

# 预测所有测试结点

dic={0:"正常用户",1:"欺诈用户"}
node_idx = 0
y_pred = predict(data, node_idx)
print(y_pred)
print(f'节点 {node_idx} 预测对应的标签为:{torch.argmax(y_pred)}, 为{dic[torch.argmax(y_pred).item()]}。')

node_idx = 1
y_pred = predict(data, node_idx)
print(y_pred)
print(f'节点 {node_idx} 预测对应的标签为:{torch.argmax(y_pred)}, 为{dic[torch.argmax(y_pred).item()]}。')

# from utils import DGraphFin
# from utils.evaluator import Evaluator
# import torch
# import torch.nn.functional as F
# import torch.nn as nn
# import torch_geometric.transforms as T
# from torch_geometric.data import Data
# import numpy as np
# import os

# def predict(data,node_id):
#     """
#     加载模型和模型预测
#     :param node_id: int, 需要进行预测节点的下标
#     :return: tensor, 类0以及类1的概率, torch.size[1,2]
#     """
#     # 这里可以加载你的模型
#     model = 
#     model.load_state_dict(torch.load('./results/model.pt'))
#     # 模型预测时，测试数据已经进行了归一化处理
#     # -------------------------- 实现模型预测部分的代码 ---------------------------

    
#     return y_pred



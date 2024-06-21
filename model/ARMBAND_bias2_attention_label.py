import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from torch.nn.utils import weight_norm
from torch.autograd import Variable

#torch.manual_seed(0)
#torch.cuda.manual_seed(0)
warnings.filterwarnings("error")
initializer = nn.init.xavier_uniform_
################
# input data shape : (batch_size, 12, 8, 7)
################
adj = torch.ones((8,8))


#modified version for dismiss some transpose and 
class GNN(nn.Module):
    def __init__(self, input_feat, output_feat, indicator):
        super(GNN, self).__init__()
        self.W_gnn = nn.Parameter(initializer(torch.randn(100, 100))) ##   # gnn 이랑 attention인 경우에 사용
        self.W_gnn2 = nn.Parameter(initializer(torch.randn(input_feat * 7, output_feat * 7)))  # gnn 이랑 attention인 경우에 사용
        self.B_gnn = nn.Parameter((torch.randn(100))) ##
        self.W_cat = nn.Parameter(initializer(torch.randn(input_feat *7*2, output_feat*7)))  # concat만 할 경우 사용
        self.B_cat = nn.Parameter((torch.randn(output_feat * 7)))
        self.W_att = nn.Parameter(initializer(torch.randn(output_feat* 2*7, 1)))   #W_att 에 쓰이는 파라미터
        self.W_att2 = nn.Parameter(initializer(torch.randn(output_feat * 2 * 7, 1)))
        self.W_att3 = nn.Parameter(initializer(torch.randn(output_feat * 2 * 7, 1)))
        self.MHA = torch.nn.MultiheadAttention(embed_dim=100, num_heads=4, batch_first=True)##


        ### for GAT --0705
        self.W_head1 = nn.Parameter(initializer(torch.randn(input_feat * 7, output_feat*7 // 4)))
        self.W_head2 = nn.Parameter(initializer(torch.randn(input_feat * 7, output_feat * 7 // 4)))
        self.W_head3 = nn.Parameter(initializer(torch.randn(input_feat * 7, output_feat * 7 // 4)))
        self.W_head4 = nn.Parameter(initializer(torch.randn(input_feat * 7, output_feat * 7 // 4)))
        self.W_alpha1 = nn.Parameter(initializer(torch.randn(output_feat*7//2, 1)))
        self.W_alpha2 = nn.Parameter(initializer(torch.randn(output_feat * 7 // 2, 1)))
        self.W_alpha3 = nn.Parameter(initializer(torch.randn(output_feat * 7 // 2, 1)))
        self.W_alpha4 = nn.Parameter(initializer(torch.randn(output_feat * 7 // 2, 1)))


        self.indicator = indicator
        self.output_feat = output_feat
        
        self.attention_storage = {}


    def forward(self, x, labels=None):
        B , C, T= x.shape   # B: batch size, T: time, C : channel(8), F : features
        a, b = self.MHA(x, x, x)

        if self.indicator == 0:   #GCN
            adj2 = torch.ones((B, 8, 8))/8
            print(adj2.shape, self.B_gnn.shape)
            x = torch.bmm(adj2+self.B_gnn, x)
            x = torch.matmul(x, self.W_gnn)
            # return x
            # x += self.B_gnn

        #default indicator(args.type) == 2
        else:   #attention
            x = torch.bmm(b, x)
            x = torch.matmul(x, self.W_gnn)
            x += self.B_gnn
            
        if labels is not None:
            for idx, label in enumerate(labels):
                label = label.item()
                if label not in self.attention_storage:
                    self.attention_storage[label] = []
                self.attention_storage[label].append(b[idx].detach())  # Store attention weights for the given label

 
        return x#torch.nn.functional.relu(x)
    
    def get_average_attention(self, label):
        """Retrieve the average attention matrix for a given label."""
        if label not in self.attention_storage:
            raise ValueError(f"No attention weights stored for label {label}.")

        # Compute the average attention weights across all stored matrices for the label
        average_attention = torch.mean(torch.stack(self.attention_storage[label]), dim=0)
        
        return average_attention
        





class Temporal_layer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Temporal_layer, self).__init__()
        self.WT_input = torch.nn.Parameter(initializer(torch.randn(32, 32, 1, in_dim-out_dim+1)))  #out_dim, in_dim, 1, 1)))
        self.WT_glu = torch.nn.Parameter(initializer(torch.randn(64, 32, 1, in_dim-out_dim+1)))  #out_dim*2, in_dim, 1, 1)))
        self.B_input = nn.Parameter((torch.FloatTensor(7)))
        self.B_glu = nn.Parameter((torch.FloatTensor(14)))
        self.out_dim = out_dim
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        x_input = F.conv2d(x, self.WT_input)#, bias = self.B_input)
        x_glu = F.conv2d(x, self.WT_glu)#, bias=self.B_glu)
        # print(x_input.shape, x_glu.shape, "input's shape")
        return (x_glu[:,0:32,:,:]+x_input)*self.sigmoid(x_glu[:,-32:,:,:])
    

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:,:,:-self.chomp_size].contiguous()   
    
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        '''
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        '''
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
    
class Spatial_layer(nn.Module):
    def __init__(self, in_dim, out_dim, indicator):
        super(Spatial_layer, self).__init__()
        self.WS_input = torch.nn.Parameter(initializer(torch.randn(out_dim, in_dim, 1, 1)))
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.batch1 = nn.BatchNorm1d(in_dim)
        self.gnn = GNN(in_dim, out_dim, indicator)
        self.gnn2 = GNN(in_dim, out_dim, indicator)
        self.batch2 = nn.BatchNorm1d(in_dim)
        
        #self.labels = labels
    def forward(self, x):
        x2 = self.gnn(x)
        self.batch1(x2)
        x2 = F.relu(x2)

        x2 = self.gnn2(x2)
        x2 = self.batch2(x2)
        x2 = F.relu(x2)

        return x+x2
    
    
class Spatial_layer2(nn.Module):
    def __init__(self, in_dim, out_dim, indicator):
        super(Spatial_layer2, self).__init__()
        self.WS_input = torch.nn.Parameter(initializer(torch.randn(out_dim, in_dim, 1, 1)))
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.batch1 = nn.BatchNorm1d(in_dim)
        self.gnn = GNN(in_dim, out_dim, indicator)
        self.gnn2 = GNN(in_dim, out_dim, indicator)
        self.batch2 = nn.BatchNorm1d(in_dim)
    def forward(self, x, labels):
        x2 = self.gnn(x, labels)
        self.batch1(x2)
        x2 = F.relu(x2)
        '''
        x2 = self.gnn2(x2)
        x2 = self.batch2(x2)
        x2 = F.relu(x2)

        '''
        
        
        return x+x2
    

def normalize_tensor_batch(tensor_batch, axis=1):
    mean = tensor_batch.mean(axis, keepdim=True)
    std = tensor_batch.std(axis, keepdim=True) + 1e-6
    normalized_batch = (tensor_batch - mean) / std
    return normalized_batch




####### baseline S2RNN  Domain_adaptation_pre
class Domain_adaptation_pre(nn.Module):
    def __init__(self, electrode_num):
        super(Domain_adaptation_pre, self).__init__()
        self.Linear = nn.Linear(100, 100)
    def forward(self, x):
        
        if(x.shape[2] != 100):
            x = x.transpose(1, 2)
        x = self.Linear(x)        
        
        return x
    
class Domain_adaptation_mlp(nn.Module):
    def __init__(self, basic_model, electrode_num):
        super(Domain_adaptation_mlp, self).__init__()
        self.basic_model = basic_model
        self.MLP1 = nn.Linear(100, 100)
        self.MLP2 = nn.Linear(300,100)
        self.MLP3 = nn.Linear(100, 100)
        self.drop1 = nn.Dropout2d(p=0.1)
        self.drop2 = nn.Dropout2d(p=0.01)
        
        self.Spat = Spatial_layer2(electrode_num, electrode_num, 2)#(second, second, indicator)
        

    def forward(self, x):
        
        if(x.shape[2] != 100):
            x = x.transpose(1, 2)
        
        x = self.MLP1(x)        
        
        pred = self.basic_model(x)
        
        return pred
    
class S2RNN(nn.Module):
    def __init__(self, channel, num_label):
        super(S2RNN, self).__init__()
        self.lstm = nn.LSTM(input_size=channel, hidden_size=512,
                            num_layers=2, batch_first=True, dropout=0.5)
        self.linear = nn.Linear(512, num_label)
    def forward(self, x):
        if(x.shape[2] != 100):
            x = x.transpose(1, 2)
        x = torch.transpose(x, 1, 2)
        h_0 = Variable(torch.zeros(2, x.size(0), 512)).to(x.device) #hidden state
        c_0 = Variable(torch.zeros(2, x.size(0), 512)).to(x.device)

        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        x = self.linear(output[:,-1,:])
        return F.log_softmax(x, dim=1)
######### baseline S2RNN


    
#현재 사용하고 있는 DA, mlp 아님 gnn 사용하는 da layer
class DA_mlp_whole(nn.Module):
    def __init__(self, basic_model, electrode_num, channel):
        super(DA_mlp_whole, self).__init__()
        self.basic_model = basic_model
        self.MLP1 = nn.Linear(100, 300)
        self.MLP2 = nn.Linear(300,100)
        self.MLP3 = nn.Linear(100, 100)
        self.drop1 = nn.Dropout2d(p=0.1)
        self.drop2 = nn.Dropout2d(p=0.01)
        
        self.Spat = Spatial_layer2(electrode_num, electrode_num, 2)#(second, second, indicator)
        

    def forward(self, x, labels):
        
        if(x.shape[2] != 100):
            x = x.transpose(1, 2)
        x = self.Spat(x, labels)
        
        pred = self.basic_model(x)
        
        return pred

#현재 사용하고 있는 모델 gnn-rnn with raw data 
class ARMBANDGNN_modified_rnn_raw(nn.Module):
    def __init__(self, electrode_num, channels, num_classes, input_feature_dim):
        super(ARMBANDGNN_modified_rnn_raw, self).__init__()
        first, second, third, fourth = channels
        self.batch1 = nn.BatchNorm1d(100)#(second)
        self.Spat1 = Spatial_layer(electrode_num, electrode_num, 2)#(second, second, indicator)
        
        self.batch2 = nn.BatchNorm1d(100)#(third)
        self.batch3 = nn.BatchNorm1d(100)#(fourth)
        self.MLP1 = nn.Linear(128, 24)
        self.MLP2 = nn.Linear(700,100)
        self.MLP3 = nn.Linear(2000, num_classes)
        self.drop1 = nn.Dropout2d(p=0.4)
        self.drop2 = nn.Dropout2d(p=0.4)
        self.drop3 = nn.Dropout2d(p=0.4)
        
        self.lstm1 = nn.LSTM(input_size=electrode_num, hidden_size=128, 
                            num_layers=2, batch_first=True, dropout=0.3)
        self.lstm2 = nn.LSTM(input_size=electrode_num, hidden_size=512, #proj_size=512, 
                            num_layers=2, batch_first=True, dropout=0.5)
        
        self.linear = nn.Linear(512, num_classes)

        
    def forward(self, x):
        
        #input shape B, C, T   [b, 8, 100]

        if(x.shape[2] != 100): # for data formation for different dataset
            x = x.transpose(1, 2)
  
        x = self.Spat1(x)   #bct
        s1 = x.clone()
        
        h_1 = Variable(torch.zeros(2, x.size(0), 512)).to(x.device) #hidden state
        c_1 = Variable(torch.zeros(2, x.size(0), 512)).to(x.device)
        x = torch.transpose(x, 1, 2)
        x, (hn, cn) = self.lstm2(x, (h_1, c_1))
        #x_temp = self.MLP1(x)        
        x = self.batch2(x) #btc
        t2 = x

        x = self.linear(x[:, -1, :])
        #bs, _, _ = x.shape
        #x = x.reshape(bs, -1)
        pred = F.log_softmax(x, dim=1)
        
        z_s1 = normalize_tensor_batch(s1)
        z_t2 = normalize_tensor_batch(t2)


        return z_s1, z_t2, pred



#현재 사용하고 있는 DA, mlp 아님 gnn 사용하는 da layer
class DA_gnn_invariance(nn.Module):
    def __init__(self, basic_model, electrode_num, channel):
        super(DA_gnn_invariance, self).__init__()
        self.basic_model = basic_model
        self.MLP1 = nn.Linear(100, 300)
        self.MLP2 = nn.Linear(300,100)
        self.MLP3 = nn.Linear(100, 100)
        self.drop1 = nn.Dropout2d(p=0.1)
        self.drop2 = nn.Dropout2d(p=0.01)
        self.Spat = Spatial_layer2(electrode_num, electrode_num, 2)#(second, second, indicator)
    def forward(self, x, labels):
        
        if(x.shape[2] != 100):
            x = x.transpose(1, 2)
        x2 = self.Spat(x, labels)
        z_s1, z_t2, pred = self.basic_model(x2)
        inv_1 = z_s1
        
        z_s1, z_t2, pred = self.basic_model(x)
        inv_2 = z_s1
        return inv_1, inv_2, z_s1, z_t2, pred

class DA_gnn_invariance_ver2(nn.Module):
    def __init__(self, basic_model, electrode_num, channel):
        super(DA_gnn_invariance_ver2, self).__init__()
        self.basic_model = basic_model
        self.MLP1 = nn.Linear(100, 300)
        self.MLP2 = nn.Linear(300,100)
        self.MLP3 = nn.Linear(100, 100)
        self.drop1 = nn.Dropout2d(p=0.1)
        self.drop2 = nn.Dropout2d(p=0.01)
        self.Spat = Spatial_layer2(electrode_num, electrode_num, 2)#(second, second, indicator)
    def forward(self, x, labels):
        
        if(x.shape[2] != 100):
            x = x.transpose(1, 2)
        x2 = self.Spat(x, labels)
        #em_da = normalize_tensor_batch(x2)
        em_da = x2
        z_s1, z_t2, pred = self.basic_model(x2)
        inv_1 = z_s1
        
        z_s1, z_t2, pred = self.basic_model(x)
        inv_2 = z_s1
        return inv_1, inv_2, z_s1, z_t2, em_da, pred

class DA_gnn_invariance_ver3(nn.Module):
    def __init__(self, basic_model, electrode_num, channel):
        super(DA_gnn_invariance_ver3, self).__init__()
        self.basic_model = basic_model
        self.MLP1 = nn.Linear(100, 300)
        self.MLP2 = nn.Linear(300,100)
        self.MLP3 = nn.Linear(100, 100)
        self.drop1 = nn.Dropout2d(p=0.1)
        self.drop2 = nn.Dropout2d(p=0.01)
        self.Spat = Spatial_layer2(electrode_num, electrode_num, 2)#(second, second, indicator)
    def forward(self, x, labels):
        
        if(x.shape[2] != 100):
            x = x.transpose(1, 2)
        x2 = self.Spat(x, labels)
        em_da = normalize_tensor_batch(x2)
        z_s1, z_t2, pred = self.basic_model(x2)
        with_da  = z_t2
        inv_1 = z_s1
        
        z_s1, z_t2, pred = self.basic_model(x)
        inv_2 = z_s1
        without_da = z_t2
        return inv_1, inv_2, z_s1, z_t2, with_da, without_da, pred





####################################이하 사용 안 하고 있는 model들########################################################################
#tcn with raw data
class ARMBANDGNN_modified_tcn_raw(nn.Module):
    def __init__(self, electrode_num, channels, num_classes, input_feature_dim):
        super(ARMBANDGNN_modified_tcn_raw, self).__init__()
        first, second, third, fourth = channels
        self.batch1 = nn.BatchNorm1d(100)#(second)
        self.Spat1 = Spatial_layer(8, 8, 2)#(second, second, indicator)
        
        self.batch2 = nn.BatchNorm1d(100)#(third)
        self.batch3 = nn.BatchNorm1d(100)#(fourth)
        self.MLP1 = nn.Linear(fourth*100, 500)
        self.MLP2 = nn.Linear(500,2000)
        self.MLP3 = nn.Linear(2000, num_classes)
        self.drop1 = nn.Dropout2d(p=0.4)
        self.drop2 = nn.Dropout2d(p=0.4)
        self.drop3 = nn.Dropout2d(p=0.4)

        self.CNN = nn.Conv2d(16, 16, 3, padding=1)
        self.Temp1 = TemporalBlock(8, 8, 3, stride=1, dilation=1, padding=2, dropout=0.05)
        self.Temp2 = TemporalBlock(8, 32, 3, stride=1, dilation=2, padding=4, dropout=0.05)
        self.Temp3 = TemporalBlock(32, 32, 3, stride=1, dilation=4, padding=8, dropout=0.05)
        self.Temp4 = TemporalBlock(32, 32, 3, stride=1, dilation=8, padding=16, dropout=0.05)

        
    def forward(self, x):
        
        #input shape B, C, T   [256, 8, 100]
        
        B, C, T = x.shape
        x = self.Temp1(x) #bct
        x = torch.transpose(x, 1, 2)
        x = self.batch1(x) #btc
        x_temp = x
        t1 = x_temp
        
        x = torch.transpose(x, 1, 2)

  
        x = self.Spat1(x)   
        x_spatial = x
        s1 = x_spatial

        x = self.Temp2(x)

        x = torch.transpose(x, 1, 2)
        x = self.batch2(x) #btc
        x_temp = x
        t2 = x_temp


        x = torch.transpose(x, 1, 2)
        x = self.Temp3(x) #bct
        x = torch.transpose(x, 1, 2)
        x = self.batch3(x)
        x_temp = x
        t3 = x_temp
        
        x = torch.transpose(x, 1, 2)
        x = self.Temp4(x)
        
        
        bs, _, _ = x.shape
        x = x.reshape(bs, -1)
        x = self.MLP1(x)        
        x = F.relu(x)
        x = self.drop1(x)
        x = self.MLP2(x)
        x = F.relu(x)
        x = self.drop2(x)
        x = self.MLP3(x)
        pred = F.log_softmax(x, dim=1)

        z_s1 = normalize_tensor_batch(s1)
        z_t1 = normalize_tensor_batch(t1)
        z_t2 = normalize_tensor_batch(t2)
        z_t3 = normalize_tensor_batch(t3)


        return z_s1, z_t1, z_t2, z_t3, pred




class ARMBANDGNN_modified_ver4_1(nn.Module):
    def __init__(self, channels, indicator, num_classes, device):
        super(ARMBANDGNN_modified_ver4_1, self).__init__()
        self.device = device
        first, second, third, fourth = channels
        self.batch1 = nn.BatchNorm2d(second)#(second)
        self.Spat1 = Spatial_layer(second, second, indicator)#(second, second, indicator)
        
        self.batch2 = nn.BatchNorm2d(third)#(third)
        self.batch3 = nn.BatchNorm2d(fourth)#(fourth)
        self.MLP1 = nn.Linear(fourth*100*8, 512)
        self.MLP2 = nn.Linear(512,128)
        self.MLP3 = nn.Linear(128, num_classes)
        self.drop1 = nn.Dropout2d(p=0.4)
        self.drop2 = nn.Dropout2d(p=0.4)
        self.drop3 = nn.Dropout2d(p=0.4)

        self.CNN = nn.Conv2d(16, 16, 3, padding=1)
        self.Temp1 = TemporalBlock(first, second, 3, stride=1, dilation=1, padding=2, dropout=0.05)
        self.Temp2 = TemporalBlock(second, third, 3, stride=1, dilation=2, padding=4, dropout=0.05)
        self.Temp3 = TemporalBlock(third, fourth, 3, stride=1, dilation=4, padding=8, dropout=0.05)
        self.Temp4 = TemporalBlock(fourth, fourth, 3, stride=1, dilation=8, padding=16, dropout=0.05)

        
    def forward(self, x):
        
        x = x.to(self.device)
        #input shape B, T, C, F [256, 24, 8, 7]
        
        x = torch.transpose(x, 1, 2)
        x = torch.transpose(x, 2, 3)
        B, C, f, T = x.shape
        x = x.reshape(-1, f, T) 
        x = self.Temp1(x)
        x = x.reshape(B, C, -1, T)
        x = torch.transpose(x, 1, 2)
        x = self.batch1(x)
        x_temp = x
        B , _, C, T = x_temp.shape
        x_temp = torch.transpose(x_temp, 1, 3)        
        t1 = x_temp.reshape(B, T, -1)
        

  
        x = self.Spat1(x)   
        x_spatial = x
        B , f, C, T = x_spatial.shape
        x_spatial = torch.transpose(x_spatial, 1, 2)
        s1 = x_spatial.reshape(B, C, -1)

        x = torch.transpose(x, 1, 2)
        B, C, f, T = x.shape
        x = x.reshape(-1,f, T) 
        x = self.Temp2(x)
        x = x.reshape(B, C, -1, T)

        x = torch.transpose(x, 1, 2)
        x = self.batch2(x)
        x_temp = x
        B , f, C, T = x_temp.shape        
        x_temp = torch.transpose(x_temp, 1, 3)        
        t2 = x_temp.reshape(B, T, -1)


        x = torch.transpose(x, 1, 2)
        x = x.reshape(-1,f, T) 
        x = self.Temp3(x)
        BC, f, T = x.shape
        x = x.reshape(B, C, -1, T)
        x = torch.transpose(x, 1, 2)
        #x = self.batch3(x)
        x_temp = x
        B , f, C, T = x_temp.shape
        x_temp = torch.transpose(x_temp, 1, 3)
        t3 = x_temp.reshape(B, T, -1)   
        
        x = torch.transpose(x, 1, 2)
        x = x.reshape(-1,f, T) 
        x = self.Temp4(x)
        BC, f, T = x.shape
        x = x.reshape(B, C, -1, T)
        x = torch.transpose(x, 1, 2)
        
        
        bs, _, _, _ = x.shape
        x = x.reshape(bs, -1)
        x = self.MLP1(x)        
        x = F.relu(x)
        x = self.drop1(x)
        x = self.MLP2(x)
        x = F.relu(x)
        x = self.drop2(x)
        x = self.MLP3(x)
        pred = F.log_softmax(x, dim=1)

        z_s1 = normalize_tensor_batch(s1)
        z_t1 = normalize_tensor_batch(t1)
        z_t2 = normalize_tensor_batch(t2)
        z_t3 = normalize_tensor_batch(t3)


        return z_s1, z_t1, z_t2, z_t3, pred
    

class ARMBANDGNN_modified_tcn_firm(nn.Module):
    def __init__(self, channels, indicator, num_classes, device):
        super(ARMBANDGNN_modified_tcn_firm, self).__init__()
        self.device = device
        first, second, third, fourth = channels
        self.batch1 = nn.BatchNorm2d(second)#(second)
        self.Spat1 = Spatial_layer(second, second, indicator)#(second, second, indicator)
        
        self.batch2 = nn.BatchNorm2d(third)#(third)
        self.batch3 = nn.BatchNorm2d(fourth)#(fourth)
        self.MLP1 = nn.Linear(fourth*100*8, 500)
        self.MLP2 = nn.Linear(500,2000)
        self.MLP3 = nn.Linear(2000, num_classes)
        self.drop1 = nn.Dropout2d(p=0.4)
        self.drop2 = nn.Dropout2d(p=0.4)
        self.drop3 = nn.Dropout2d(p=0.4)

        self.CNN = nn.Conv2d(16, 16, 3, padding=1)
        self.Temp1 = TemporalBlock(32, 32, 3, stride=1, dilation=1, padding=2, dropout=0.05)
        self.Temp2 = TemporalBlock(32, 32, 3, stride=1, dilation=2, padding=4, dropout=0.05)
        self.Temp3 = TemporalBlock(32, 64, 3, stride=1, dilation=4, padding=8, dropout=0.05)
        self.Temp4 = TemporalBlock(64, 64, 3, stride=1, dilation=8, padding=16, dropout=0.05)

        
    def forward(self, x):
        
        x = x.to(self.device)
        #input shape B, T, C, F [256, 24, 8, 7]
        
        x = torch.transpose(x, 1, 2)
        x = torch.transpose(x, 2, 3)
        B, C, f, T = x.shape
        x = x.reshape(-1, f, T) 
        x = self.Temp1(x)
        x = x.reshape(B, C, -1, T)
        x = torch.transpose(x, 1, 2)
        x = self.batch1(x)
        x_temp = x
        B , _, C, T = x_temp.shape
        x_temp = torch.transpose(x_temp, 1, 3)        
        t1 = x_temp.reshape(B, T, -1)
        

  
        x = self.Spat1(x)   
        x_spatial = x
        B , f, C, T = x_spatial.shape
        x_spatial = torch.transpose(x_spatial, 1, 2)
        s1 = x_spatial.reshape(B, C, -1)

        x = torch.transpose(x, 1, 2)
        B, C, f, T = x.shape
        x = x.reshape(-1,f, T) 
        x = self.Temp2(x)
        x = x.reshape(B, C, -1, T)

        x = torch.transpose(x, 1, 2)
        x = self.batch2(x)
        x_temp = x
        B , f, C, T = x_temp.shape        
        x_temp = torch.transpose(x_temp, 1, 3)        
        t2 = x_temp.reshape(B, T, -1)


        x = torch.transpose(x, 1, 2)
        x = x.reshape(-1,f, T) 
        x = self.Temp3(x)
        BC, f, T = x.shape
        x = x.reshape(B, C, -1, T)
        x = torch.transpose(x, 1, 2)
        #x = self.batch3(x)
        x_temp = x
        B , f, C, T = x_temp.shape
        x_temp = torch.transpose(x_temp, 1, 3)
        t3 = x_temp.reshape(B, T, -1)   
        
        x = torch.transpose(x, 1, 2)
        x = x.reshape(-1,f, T) 
        x = self.Temp4(x)
        BC, f, T = x.shape
        x = x.reshape(B, C, -1, T)
        x = torch.transpose(x, 1, 2)
        
        
        bs, _, _, _ = x.shape
        x = x.reshape(bs, -1)
        x = self.MLP1(x)        
        x = F.relu(x)
        x = self.drop1(x)
        x = self.MLP2(x)
        x = F.relu(x)
        x = self.drop2(x)
        x = self.MLP3(x)
        pred = F.log_softmax(x, dim=1)

        z_s1 = normalize_tensor_batch(s1)
        z_t1 = normalize_tensor_batch(t1)
        z_t2 = normalize_tensor_batch(t2)
        z_t3 = normalize_tensor_batch(t3)


        return z_s1, z_t1, z_t2, z_t3, pred


class ARMBANDGNN_modified_ver2_1(nn.Module):
    def __init__(self, channels, indicator, num_classes, device):
        super(ARMBANDGNN_modified_ver2_1, self).__init__()
        self.device = device
        first, second, third, fourth = channels
        self.Temp1 = Temporal_layer(first, second)
        self.batch1 = nn.BatchNorm2d(32)#(second)
        self.Spat1 = Spatial_layer(second, second, indicator)#(second, second, indicator)
        ##
        self.Spat3 = Spatial_layer(first, first, indicator)


        self.Temp2 = Temporal_layer(second, third)
        self.batch2 = nn.BatchNorm2d(32)#(third)
        self.Temp3 = Temporal_layer(third, fourth)
        self.MLP1 = nn.Linear(fourth*8*32,500) #over fitting  # L2 reg 추가
        self.MLP2 = nn.Linear(500, 2000)
        self.MLP3 = nn.Linear(2000, num_classes)
        self.drop1 = nn.Dropout2d(p=0.4)
        self.drop2 = nn.Dropout2d(p=0.4)
        self.drop3 = nn.Dropout2d(p=0.4)
        self.MLP = nn.Linear(16*8*7, 16*8*7)
        self.CNN = nn.Conv2d(16, 16, 3, padding=1)
        
    def forward(self, x):
        
        x = x.to(self.device)
 
        x = torch.transpose(x, 1, 3)
        x = self.Temp1(x)
        
        x = self.batch1(x)
        x = torch.transpose(x, 1, 3)    
        x_temp = x
        B , T, C, _ = x_temp.shape
        t1 = x_temp.reshape(B, T, -1)   
        
        
        x = self.Spat1(x)   
        x_spatial = x
        B , T, C, _ = x_spatial.shape
        x_spatial = torch.transpose(x_spatial, 1, 2)
        s1 = x_spatial.reshape(B, C, -1)

        x = torch.transpose(x, 1, 3)       
        x = self.Temp2(x)
        x = self.batch2(x)
        x_temp = torch.transpose(x, 1, 3)    
        B , T, C, _ = x_temp.shape
        t2 = x_temp.reshape(B, T, -1)

        x = self.Temp3(x)
        x_temp = torch.transpose(x, 1, 3)    
        B , T, C, _ = x_temp.shape
        t3 = x_temp.reshape(B, T, -1) 
        
        bs, _, _, _ = x.shape
        x = x.reshape(bs, -1)
        x = self.MLP1(x)
        x = F.relu(x)
        x = self.drop1(x)
        x = self.MLP2(x)
        x = F.relu(x)
        x = self.drop2(x)
        x = self.MLP3(x)
        pred = F.log_softmax(x, dim=1)

        z_s1 = normalize_tensor_batch(s1)
        z_t1 = normalize_tensor_batch(t1)
        z_t2 = normalize_tensor_batch(t2)
        z_t3 = normalize_tensor_batch(t3)
        #print(z_s1.shape, z_t1.shape, z_t2.shape, z_t3.shape)


        return z_s1, z_t1, z_t2, z_t3, pred
    
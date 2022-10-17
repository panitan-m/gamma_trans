from audioop import add
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel

from .losses import LOSS_FN

class AddBeta(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.beta = nn.Parameter(torch.zeros(size), requires_grad=True)
    
    def forward(self, x):
        return x + self.beta
    
    def extra_repr(self):
        size_str = 'x'.join(str(size) for size in self.beta.size())
        device_str = '' if not self.beta.is_cuda else ' (GPU {})'.format(self.beta.get_device())
        return 'Parameter containing: [{} of size {}{}]'.format(
                torch.typename(self.beta), size_str, device_str)

class G_Guass(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        init_values = [0., 1., 0., 0., 0., 0., 1., 0., 0., 0.]
        self.a = nn.ParameterList([self.wi(v) for v in init_values])
        
    def wi(self, init):
        if init == 1:
            return nn.Parameter(torch.ones(self.size), requires_grad=True)
        elif init == 0:
            return nn.Parameter(torch.zeros(self.size), requires_grad=True)
        else:
            raise ValueError("Invalid argument '%d' provided for init in G_Gauss layer" % init)
        
    def forward(self, x):
        z_c, u = x
        
        def compute(y):
            return y[0] * torch.sigmoid(y[1]*u + y[2]) + y[3]*u + y[4]
        
        mu = compute(self.a[:5])
        v = compute(self.a[5:])
        
        z_est = (z_c - mu) * v + mu
        return z_est
    
    def extra_repr(self):
        return '{}'.format(self.size*10)
        
def add_noise(inputs, noise_std):
    return inputs + torch.normal(0, noise_std, size=inputs.size()).cuda()

class LadderNetworkFC(nn.Module):
    def __init__(self, args=None,
                 layer_sizes=[784, 1000, 500, 250, 250, 250, 10],
                 noise_std=0.3,
                 denoising_cost=[1000.0, 10.0, 0.10, 0.10, 0.10, 0.10, 0.10],
                 bert_checkpoint=None,
                 binary=False):
        super().__init__()
        
        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes) -1
        self.noise_std = noise_std
        self.denoising_cost = denoising_cost
        self.binary = binary
        
        if hasattr(args, 'bert'):
            self.bert = self.bert = AutoModel.from_pretrained(bert_checkpoint) if bert_checkpoint else AutoModel.from_pretrained(args.bert)
            if args.freeze_bert is not None:
                for name, param in self.bert.named_parameters():
                    if args.freeze_bert not in name:
                        param.requires_grad = False
                    else:
                        break
                    
        self.fc_enc = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=False) for i in range(self.L)])
        self.fc_dec = nn.ModuleList([nn.Linear(layer_sizes[i+1], layer_sizes[i], bias=False) for i in range(self.L)])
        self.batch_norm_enc = nn.ModuleList([nn.BatchNorm1d(s, affine=False, eps=1e-10) for s in layer_sizes[1:]])
        self.batch_norm_dec = nn.ModuleList([nn.BatchNorm1d(s, affine=False, eps=1e-10) for s in layer_sizes])
        self.betas = nn.ModuleList([AddBeta(s) for s in layer_sizes[1:]])
        self.g_guass = nn.ModuleList([G_Guass(s) for s in layer_sizes])
        
        if hasattr(args, 'loss_fn'):
            self.loss_fn = LOSS_FN[args.loss_fn]
        
    def encoder(self, inputs, noise_std):
        if hasattr(self, 'bert'):
            x, mask = inputs
            inputs = self.bert(x, attention_mask=mask)[1]
        h = add_noise(inputs, noise_std)
        all_z = [None for _ in range(len(self.layer_sizes))]
        all_z[0] = h
        
        for l in range(1, self.L+1):
            z_pre = self.fc_enc[l-1](h)
            z = self.batch_norm_enc[l-1](z_pre)
            z = add_noise(z, noise_std)
            
            if l == self.L:
                if hasattr(self, 'bert'):
                    h = self.betas[l-1](z)
                    if self.binary:
                        h = torch.sigmoid(h)
                    # h = torch.clamp(self.betas[l-1](z), 0, 1)
                else:
                    h = torch.softmax(self.betas[l-1](z), 1)
            else:
                h = torch.relu(self.betas[l-1](z))
                
            all_z[l] = z
            
        return h, all_z
    
    def forward(self, inputs_l, targets_l, inputs_u=None, epochs=None, epoch=None, encoder_only=False):
        y_c_l, _ = self.encoder(inputs_l, self.noise_std)
        y_l, _ = self.encoder(inputs_l, 0.0)
        
        if encoder_only:
            y_l = y_l.squeeze(1)
            cost = F.mse_loss(y_l, targets_l)
            return cost
        
        if self.training and inputs_u is not None:
            y_c_u, corr_z = self.encoder(inputs_u, self.noise_std)
            y_u, clean_z = self.encoder(inputs_u, 0.0)
        
            d_cost = []
            
            for l in range(self.L, -1, -1):
                z, z_c = clean_z[l], corr_z[l]
                if l == self.L:
                    u = y_c_u
                else:
                    u = self.fc_dec[l](z_est)
                u = self.batch_norm_dec[l](u)
                z_est = self.g_guass[l]([z_c, u])
                d_cost.append((torch.mean(torch.sum(torch.square(z_est - z), 1)) / self.layer_sizes[l]) * self.denoising_cost[l])
            
            u_cost = sum(d_cost)
            
            if hasattr(self, 'bert'):
                y_c_l = y_c_l
                # x_cost = F.mse_loss(y_c_l, targets_l)
                x_cost = self.loss_fn(y_c_l, targets_l, epochs=epochs, epoch=epoch)
                pred = y_c_l
            else:
                x_cost = F.cross_entropy(y_c_l, targets_l)
                _, pred = y_c_l.topk(1, 1)
                pred = pred.t().squeeze(1)
                
            cost = x_cost + u_cost
    
            return cost, u_cost, pred
        
        else:
            if hasattr(self, 'bert'):
                y_l = y_l
                cost = F.mse_loss(y_l, targets_l)
                pred = y_l
                return cost, pred
            else:
                _, pred = y_l.topk(1, 1)
                pred = pred.t().squeeze(1)
                return pred
            
            
class LadderNetworkGamma(nn.Module):
    def __init__(self, args=None,
                 output_size=1,
                 noise_std=0.3,
                 denoising_cost=1,
                 bert_checkpoint=None,
                 binary=False):
        super().__init__()
        
        self.output_size = output_size
        self.noise_std = noise_std
        self.denoising_cost = denoising_cost
        self.binary = binary
        
        if hasattr(args, 'bert'):
            self.long = True if 'long' in args.bert else False
            self.bert = self.bert = AutoModel.from_pretrained(bert_checkpoint) if bert_checkpoint else AutoModel.from_pretrained(args.bert)
            if args.freeze_bert is not None:
                for name, param in self.bert.named_parameters():
                    if args.freeze_bert not in name:
                        param.requires_grad = False
                    else:
                        break
                    
        self.fc_enc = nn.Linear(self.bert.config.hidden_size, self.output_size)
        # self.fc_dec = nn.Linear(self.output_size, self.bert.config.hidden_size)
        # self.batch_norm_enc = nn.ModuleList([nn.BatchNorm1d(s, affine=False, eps=1e-10) for s in layer_sizes[1:]])
        # self.batch_norm_dec = nn.ModuleList([nn.BatchNorm1d(s, affine=False, eps=1e-10) for s in layer_sizes])
        # self.betas = nn.ModuleList([AddBeta(s) for s in layer_sizes[1:]])
        self.g_guass = G_Guass(output_size)
        
        if hasattr(args, 'loss_fn'):
            self.loss_fn = LOSS_FN[args.loss_fn]
        
    def encoder(self, inputs, noise_std):
        
        all_z = [None for _ in range(len(self.bert.encoder.layer)+2)]
        
        if hasattr(self, 'bert'):
            input_ids, attention_mask = inputs
        device = input_ids.device
        input_shape = input_ids.size()
        
        if self.long:
            _, input_ids, attention_mask, _, _, _ = self.bert._pad_to_window_size(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=torch.zeros(input_shape, dtype=torch.long, device=device),
                position_ids=None,
                inputs_embeds=None,
                pad_token_id=self.bert.config.pad_token_id,
            )
            extended_attention_mask: torch.Tensor = self.bert.get_extended_attention_mask(attention_mask, input_shape, device)[:, 0, 0, :]
            is_index_masked = attention_mask < 0
            is_index_global_attn = attention_mask > 0
            is_global_attn = is_index_global_attn.flatten().any().item()
        else:
            extended_attention_mask: torch.Tensor = self.bert.get_extended_attention_mask(attention_mask, input_shape, device)

        embedding_output = self.bert.embeddings(input_ids=input_ids)
        h = add_noise(embedding_output, noise_std)
        
        for i, layer_module in enumerate(self.bert.encoder.layer):
            if self.long:
                layer_outputs = layer_module(
                    h, 
                    extended_attention_mask,
                    is_index_masked=is_index_masked,
                    is_index_global_attn=is_index_global_attn,
                    is_global_attn=is_global_attn,
                    )
            else:
                layer_outputs = layer_module(h, extended_attention_mask)
            h = layer_outputs[0]
            h = add_noise(h, noise_std)
            all_z[i] = h
  
        z = self.bert.pooler.dense(h[:, 0])
        z = add_noise(z, noise_std)
        h = self.bert.pooler.activation(z)
        all_z[-2] = z
        
        z = self.fc_enc(h)
        z = add_noise(z, noise_std)
        all_z[-1] = z
        
        h = z
        if self.binary:
            h = torch.sigmoid(z)
            
        return h, all_z
    
    def forward(self, inputs_l, targets_l, inputs_u=None, epochs=None, epoch=None): 
        y_c_l, _ = self.encoder(inputs_l, self.noise_std)
        y_l, _ = self.encoder(inputs_l, 0.0)
        
        if self.training and inputs_u is not None:
            y_c_u, corr_z = self.encoder(inputs_u, self.noise_std)
            y_u, clean_z = self.encoder(inputs_u, 0.0)
        
            d_cost = []
            
            z, z_c = clean_z[-1], corr_z[-1]
            u = y_c_u
                # else:
                #     u = self.fc_dec[l](z_est)
                # u = self.batch_norm_dec[l](u)
            z_est = self.g_guass([z_c, u])
            d_cost.append(torch.mean(torch.sum(torch.square(z_est - z), 1)) * self.denoising_cost)
            
            u_cost = sum(d_cost)
            
            if hasattr(self, 'bert'):
                y_c_l = y_c_l
                # x_cost = F.mse_loss(y_c_l, targets_l)
                x_cost = self.loss_fn(y_c_l, targets_l, epochs=epochs, epoch=epoch)
                pred = y_c_l
            else:
                x_cost = F.cross_entropy(y_c_l, targets_l)
                _, pred = y_c_l.topk(1, 1)
                pred = pred.t().squeeze(1)
                
            cost = x_cost + u_cost
    
            return cost, u_cost, pred
        
        else:
            if hasattr(self, 'bert'):
                y_l = y_l
                cost = self.loss_fn(y_l, targets_l)
                pred = y_l
                return cost, pred
            else:
                _, pred = y_l.topk(1, 1)
                pred = pred.t().squeeze(1)
                return pred
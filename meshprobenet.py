import torch
import torch.nn as nn
import torch.nn.functional as F

class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, n_layers, n_gpu, variable_lengths=True, rnn_type='lstm'):
        super(BiRNN, self).__init__()
        self.variable_lengths = variable_lengths
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if rnn_type == 'gru':
            self.rnn = nn.GRU(embed_dim, hidden_size, n_layers, batch_first=True, bidirectional=True)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(embed_dim, hidden_size, n_layers, batch_first=True, bidirectional=True)
        self.multigpu = n_gpu > 1

    def forward(self, input_var, input_lengths):
        if self.multigpu:
            self.rnn.flatten_parameters()
        embedded = self.embedding(input_var)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        output, hidden = self.rnn(embedded)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden
    
    
class MeSHProbes(nn.Module):    
    def __init__(self, hidden_size, n_probes):
        super(MeSHProbes, self).__init__()
        self.self_attn = nn.Parameter(torch.FloatTensor(n_probes, hidden_size))

    def forward(self, birnn_outputs):
        batch_size = birnn_outputs.size(0)
        input_size = birnn_outputs.size(1)
        # (batch, n_probes, dim) * (batch, in_len, dim) -> (batch, n_probes, in_len)
        attn = torch.bmm(self.self_attn.expand(batch_size, -1, -1), birnn_outputs.transpose(1, 2))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        # (batch, n_probes, in_len) * (batch, in_len, dim) -> (batch, n_probes, dim)
        context_vectors = torch.bmm(attn, birnn_outputs)
        # (batch, n_probes, dim) -> (batch, n_probes * dim)
        context_vectors = context_vectors.view(batch_size, -1)
        return context_vectors
    

class MultiViewC(nn.Module):
    def __init__(self, n_jrnl, jrnl_dim, mesh_size, hidden_size, n_probes):
        super(MultiViewC, self).__init__()
        self.jrnl_embedding = nn.Embedding(n_jrnl, jrnl_dim)
        self.out_mesh_dstrbtn = nn.Linear(hidden_size * n_probes + jrnl_dim, mesh_size)

    def forward(self, jrnl_variable, context_vectors):
        jrnl_embedded = self.jrnl_embedding(jrnl_variable).squeeze(dim=1)   
        combined = torch.cat((context_vectors, jrnl_embedded), dim=1)            
        output_dstrbtn = self.out_mesh_dstrbtn(combined)  
        return output_dstrbtn


class MeSHProbeNet(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, n_layers, n_probes, n_jrnl, jrnl_dim, mesh_size, n_gpu):
        super(MeSHProbeNet, self).__init__()
        self.birnn = BiRNN(vocab_size, embed_dim, hidden_size, n_layers, n_gpu)
        self.meshprobes = MeSHProbes(hidden_size * 2, n_probes) # *2 because of bidirection
        self.multiviewc = MultiViewC(n_jrnl, jrnl_dim, mesh_size, hidden_size * 2, n_probes)        
        self.loss_fct = nn.MultiLabelSoftMarginLoss()
            
    def forward(self, input_variables, input_lengths, jrnl_variables, target_variables=None):
        birnn_outputs, birnn_hidden = self.birnn(input_variables, input_lengths)
        context_vectors = self.meshprobes(birnn_outputs)   
        logits = self.multiviewc(jrnl_variables, context_vectors)
        
        if target_variables is None:
            return logits
        loss = self.loss_fct(logits, target_variables)
        return loss.view(-1)
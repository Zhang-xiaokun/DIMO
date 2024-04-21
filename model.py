import datetime
import math
import numpy as np
import torch
from torch import nn, backends
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.sparse
from scipy.sparse import coo
import time
import pickle




def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


class HyperConv(Module):
    def __init__(self, layers, emb_size, n_node):
        super(HyperConv, self).__init__()
        self.emb_size = emb_size
        self.layers = layers
        self.n_node = n_node

        self.w_co = nn.Linear(self.emb_size, self.emb_size)

        self.gate_w1 = nn.Linear(self.emb_size*2, self.emb_size)

        self.dropout10 = nn.Dropout(0.1)
        self.dropout20 = nn.Dropout(0.2)
        self.dropout30 = nn.Dropout(0.3)
        self.dropout40 = nn.Dropout(0.4)
        self.dropout50 = nn.Dropout(0.5)
        self.dropout60 = nn.Dropout(0.6)
        self.dropout70 = nn.Dropout(0.7)

    def forward(self, adjacency, embedding, mo_embedding):
        id_embeddings = self.dropout30(embedding)
        mo_embedding = self.dropout30(mo_embedding)
        for i in range(self.layers):
            item_co_emb = self.get_embedding(adjacency, id_embeddings)
            id_embeddings = id_embeddings + item_co_emb
            item_mo_emb = self.get_embedding(adjacency, mo_embedding)
            mo_embedding = mo_embedding + item_mo_emb
        results = id_embeddings
        mo_results = mo_embedding
        return results, mo_results
    def get_embedding(self, adjacency, embedding):
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)

        shape = adjacency.shape
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        # item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), embedding)
        adjacency = trans_to_cuda(adjacency)
        item_embeddings = torch.mm(adjacency.to_dense(), trans_to_cuda(embedding))
        return item_embeddings

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Disen(Module):
    def __init__(self, n_node, adjacency, lr, co_layer, l2, item_beta, pro_beta, cri_beta, dataset, num_heads=4, emb_size=100, text_emb_size=100, batch_size=100, num_negatives=100):
        super(Disen, self).__init__()
        self.emb_size = emb_size
        self.text_emb_size = text_emb_size
        self.batch_size = batch_size
        self.n_node = n_node
        self.L2 = l2
        self.lr = lr
        self.co_layer = co_layer
        self.item_beta = item_beta
        self.pro_beta = pro_beta
        self.cri_beta = cri_beta
        self.num_negatives = num_negatives
        self.num_heads = num_heads

        self.embedding = nn.Embedding(self.n_node, self.emb_size)

        self.LayerNorm = LayerNorm(self.emb_size, eps=1e-12)

        self.adjacency = adjacency
        self.HyperGraph = HyperConv(self.co_layer, self.emb_size, self.n_node)


        # introducing text embeddings
        text_path = './datasets/' + dataset + '/textMatrixpca100.npy'
        textWeights = np.load(text_path)
        self.text_embedding = nn.Embedding(self.n_node, text_emb_size)
        text_pre_weight = np.array(textWeights)
        self.text_embedding.weight.data.copy_(torch.from_numpy(text_pre_weight))

        # introducing frequence relations
        frequence = pickle.load(open('./datasets/' + dataset + '/frequence.txt', 'rb'))
        self.pos_item = np.array(frequence[0])
        self.neg_item = np.array(frequence[1])
        self.pos_weight = np.array(frequence[2])


        self.pos_embedding = nn.Embedding(2000, self.emb_size)


        self.active = nn.ReLU()
        self.relu = nn.ReLU()


        # self_attention
        if self.emb_size % num_heads != 0:  # 整除
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (emb_size, num_heads))
            # 参数定义
        # self.num_heads = num_heads  # 4
        self.attention_head_size = int(emb_size / num_heads)  # 16  每个注意力头的维度
        self.all_head_size = int(self.num_heads * self.attention_head_size)
        # query, key, value 的线性变换（上述公式2）
        self.query = nn.Linear(self.emb_size, self.emb_size)  # 128, 128
        self.key = nn.Linear(self.emb_size, self.emb_size)
        self.value = nn.Linear(self.emb_size, self.emb_size)

        self.query_id = nn.Linear(self.emb_size, self.emb_size)  # 128, 128
        self.key_id = nn.Linear(self.emb_size, self.emb_size)
        self.value_id = nn.Linear(self.emb_size, self.emb_size)

        self.dropout = nn.Dropout(0.2)
        self.emb_dropout = nn.Dropout(0.25)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout5 = nn.Dropout(0.5)
        self.dropout7 = nn.Dropout(0.7)

        self.cos_sim = nn.CosineSimilarity(dim=-1)

        self.sim_w_p1u = nn.Linear(self.emb_size, self.emb_size)
        self.sim_w_p1d = nn.Linear(self.emb_size, self.emb_size)
        self.sim_w_p2u = nn.Linear(self.emb_size, self.emb_size)
        self.sim_w_p2d = nn.Linear(self.emb_size, self.emb_size)

        self.sim_w_c1u = nn.Linear(self.emb_size, self.emb_size)
        self.sim_w_c2u = nn.Linear(self.emb_size, self.emb_size)
        self.sim_w_c1d = nn.Linear(self.emb_size, self.emb_size)
        self.sim_w_c2d = nn.Linear(self.emb_size, self.emb_size)

        self.sim_w_i1 = nn.Linear(self.emb_size, self.emb_size)
        self.sim_w_i2 = nn.Linear(self.emb_size, self.emb_size)


        self.w_2 = nn.Linear(self.emb_size, 1)
        self.glu1 = nn.Linear(self.emb_size, self.emb_size)
        self.glu2 = nn.Linear(self.emb_size, self.emb_size, bias=False)

        self.mlp_seq1 = nn.Linear(self.emb_size, self.emb_size)
        self.mlp_seq2 = nn.Linear(self.emb_size, self.emb_size)
        self.mlp_seq3 = nn.Linear(self.emb_size, self.emb_size)
        self.mlp_seq4 = nn.Linear(self.emb_size, self.emb_size)
        self.mlp_seq5 = nn.Linear(self.emb_size, self.emb_size)
        self.mlp_seq6 = nn.Linear(self.emb_size, self.emb_size)
        self.mlp_seq7 = nn.Linear(self.emb_size, self.emb_size)
        self.mlp_seq8 = nn.Linear(self.emb_size, self.emb_size)


        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def generate_sess_emb(self, item_embedding, text_embedding, session_item, session_len, reversed_sess_item, mask):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        # zeros = torch.zeros(1, self.emb_size)
        mask = mask.float().unsqueeze(-1)

        item_emb_table = torch.cat([zeros, item_embedding], 0)
        text_emb_table = torch.cat([zeros, text_embedding], 0)

        # id seq emb
        get = lambda i: item_emb_table[session_item[i]]
        seq_h = torch.cuda.FloatTensor(self.batch_size, list(session_item.shape)[1], self.emb_size).fill_(0)
        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)
        # id proxy
        id_sess_pro_emb = torch.div(torch.sum(seq_h, 1), session_len.type(torch.cuda.FloatTensor))
        # Self-attention to get session emb
        attention_mask = mask.permute(0, 2, 1).unsqueeze(1)  # [bs, 1, 1, seqlen] 
        attention_mask = (1.0 - attention_mask) * -10000.0

        mixed_query_layer = self.query_id(seq_h)  # [bs, seqlen, hid_size]
        mixed_key_layer = self.key_id(seq_h)  # [bs, seqlen, hid_size]
        mixed_value_layer = self.value_id(seq_h)  # [bs, seqlen, hid_size]

        attention_head_size = int(self.emb_size / self.num_heads)
        query_layer = self.transpose_for_scores(mixed_query_layer, attention_head_size)  # [bs, 8, seqlen, 16]
        key_layer = self.transpose_for_scores(mixed_key_layer, attention_head_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, attention_head_size)  # [bs, 8, seqlen, 16]
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # [bs, 8, seqlen, 16]*[bs, 8, 16, seqlen]  ==> [bs, 8, seqlen, seqlen]
        attention_scores = attention_scores / math.sqrt(attention_head_size)  # [bs, 8, seqlen, seqlen]
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [bs, 8, seqlen, seqlen]
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)  # [bs, 8, seqlen, 16]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [bs, seqlen, 8, 16]
        new_context_layer_shape = context_layer.size()[:-2] + (self.emb_size,)  # [bs, seqlen, 128]
        sa_result = context_layer.view(*new_context_layer_shape)

        item_pos = torch.tensor(range(1, seq_h.size()[1] + 1), device='cuda')
        item_pos = item_pos.unsqueeze(0).expand_as(session_item)

        item_pos = item_pos * mask.squeeze(2)
        item_last_num = torch.max(item_pos, 1)[0].unsqueeze(1).expand_as(item_pos)
        last_pos_t = torch.where(item_pos - item_last_num >= 0, torch.tensor([1.0], device='cuda'),
                                 torch.tensor([0.0], device='cuda'))
        as_last_unit = last_pos_t.unsqueeze(2).expand_as(sa_result) * sa_result
        id_sess_emb = torch.sum(as_last_unit, 1)



        # modality seq emb
        get_text = lambda i: text_emb_table[session_item[i]]
        seq_text = torch.cuda.FloatTensor(self.batch_size, list(session_item.shape)[1], self.emb_size).fill_(0)
        for i in torch.arange(session_item.shape[0]):
            seq_text[i] = get_text(i)
        # modality proxy
        mo_sess_pro_emb = torch.div(torch.sum(seq_text, 1), session_len.type(torch.cuda.FloatTensor))

        # Self-Attention to obtain text seq_emb
        attention_mask = mask.permute(0, 2, 1).unsqueeze(1)  # [bs, 1, 1, seqlen] 增加维度
        attention_mask = (1.0 - attention_mask) * -10000.0

        mixed_query_layer = self.query(seq_text)  # [bs, seqlen, hid_size]
        mixed_key_layer = self.key(seq_text)  # [bs, seqlen, hid_size]
        mixed_value_layer = self.value(seq_text)  # [bs, seqlen, hid_size]

        attention_head_size = int(self.emb_size / self.num_heads)
        query_layer = self.transpose_for_scores(mixed_query_layer, attention_head_size)  # [bs, 8, seqlen, 16]
        key_layer = self.transpose_for_scores(mixed_key_layer, attention_head_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, attention_head_size)  # [bs, 8, seqlen, 16]
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # [bs, 8, seqlen, 16]*[bs, 8, 16, seqlen]  ==> [bs, 8, seqlen, seqlen]
        attention_scores = attention_scores / math.sqrt(attention_head_size)  # [bs, 8, seqlen, seqlen]
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [bs, 8, seqlen, seqlen]
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)  # [bs, 8, seqlen, 16]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [bs, seqlen, 8, 16]
        new_context_layer_shape = context_layer.size()[:-2] + (self.emb_size,)  # [bs, seqlen, 128]
        sa_result = context_layer.view(*new_context_layer_shape)
        # last hidden state as price preferences
        item_pos = torch.tensor(range(1, seq_text.size()[1] + 1), device='cuda')
        item_pos = item_pos.unsqueeze(0).expand_as(session_item)

        item_pos = item_pos * mask.squeeze(2)
        item_last_num = torch.max(item_pos, 1)[0].unsqueeze(1).expand_as(item_pos)
        last_pos_t = torch.where(item_pos - item_last_num >= 0, torch.tensor([1.0], device='cuda'),
                                 torch.tensor([0.0], device='cuda'))
        as_last_unit = last_pos_t.unsqueeze(2).expand_as(sa_result) * sa_result
        mo_sess_emb = torch.sum(as_last_unit, 1)


        return id_sess_emb, mo_sess_emb, id_sess_pro_emb, mo_sess_pro_emb

    def id_co_loss(self, item_emb, pos_item, neg_item, weights):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        # zeros = torch.zeros(1, self.emb_size)
        item_emb_table = torch.cat([zeros, item_emb], 0)
        item_batch = torch.chunk(item_emb, self.n_node//512, 0)
        pos_batch = torch.chunk(torch.tensor(pos_item), self.n_node // 512, 0)
        neg_batch = torch.chunk(torch.tensor(neg_item), self.n_node // 512, 0)
        weight_batch = torch.chunk(torch.tensor(weights), self.n_node // 512, 0)
        con_loss = 0
        tau = 1
        for id_temp_emb, pos_temp, neg_temp, weight_temp in zip(item_batch, pos_batch, neg_batch, weight_batch):
            pos_temp_emb = item_emb_table[pos_temp]
            neg_temp_emb = item_emb_table[neg_temp]
            weight_temp = -trans_to_cuda(weight_temp)
            weight_temp = weight_temp / (torch.sum(weight_temp, 1).unsqueeze(1).repeat(1,weight_temp.shape[1]) + 1e-8)
            pos_temp_emb = pos_temp_emb * weight_temp.unsqueeze(2).repeat(1,1,pos_temp_emb.shape[2])
            pos_mean_emb = torch.sum(pos_temp_emb, 1)
            pos_dis = self.cos_sim(id_temp_emb, pos_mean_emb)
            fenzi = pos_dis/tau
            fenzi = torch.exp(fenzi)
            id_temp_emb_expand = id_temp_emb.unsqueeze(1).repeat(1, neg_temp_emb.shape[1], 1)
            neg_dis = torch.exp(self.cos_sim(id_temp_emb_expand, neg_temp_emb)/tau)
            fenmu = torch.sum(neg_dis, 1)
            fenmu = fenzi + fenmu
            fenzi = torch.log10(fenzi)
            fenmu = torch.log10(fenmu)
            temp_loss = fenmu - fenzi
            temp_loss = torch.sum(temp_loss, 0)
            con_loss += temp_loss


        return con_loss

    def seq_pro_loss(self, sess_id, sess_id_pro, sess_mo, sess_mo_pro):
        s_id_loss1 = self.cos_sim(self.mlp_seq1(sess_id), self.mlp_seq2(sess_id_pro))
        s_id_loss2 = self.cos_sim(self.mlp_seq3(sess_id), self.mlp_seq4(sess_mo))


        s_id_loss = torch.log10(torch.exp(s_id_loss1)) - torch.log10(torch.exp(s_id_loss2)+torch.exp(s_id_loss1))
        s_id_loss = torch.sum(s_id_loss, 0)
        s_mo_loss1 = self.cos_sim(self.mlp_seq4(sess_mo), self.mlp_seq5(sess_mo_pro))
        s_mo_loss2 = self.cos_sim(self.mlp_seq7(sess_mo), self.mlp_seq8(sess_id))

        s_mo_loss = torch.log10(torch.exp(s_mo_loss1)) - torch.log10(torch.exp(s_mo_loss2)+torch.exp(s_mo_loss1))
        s_mo_loss = torch.sum(s_mo_loss, 0)
        loss_pro = s_id_loss + s_mo_loss
        return -loss_pro

    def seq_cri_loss(self, item_emb, text_emb, sess_id_emb, sess_mo_emb, flag, lab):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        # zeros = torch.zeros(1, self.emb_size)
        item_emb_table = torch.cat([zeros, item_emb], 0)
        text_emb_table = torch.cat([zeros, text_emb], 0)
        lab_id_emb = item_emb_table[lab]
        lab_mo_emb = text_emb_table[lab]
        id_dis = self.cos_sim(sess_id_emb, lab_id_emb)
        mo_dis = self.cos_sim(sess_mo_emb, lab_mo_emb)

        cri_id_mo = (torch.log10(torch.exp(id_dis)) - torch.log10(torch.exp(mo_dis) + torch.exp(id_dis)))*flag
        cri_loss = torch.sum(cri_id_mo,0)
        return -cri_loss

    def sim_cal_w(self, emb1, emb2, mat_w):
        sim = emb1 * mat_w(emb2)
        sim = torch.sum(sim,-1)
        return sim

    def transpose_for_scores(self, x, attention_head_size):
        # INPUT:  x'shape = [bs, seqlen, hid_size]  假设hid_size=128
        new_x_shape = x.size()[:-1] + (self.num_heads, attention_head_size)  # [bs, seqlen, 8, 16]
        x = x.view(*new_x_shape)  #
        return x.permute(0, 2, 1, 3)


    def forward(self, session_item, flag, session_len, reversed_sess_item, mask, target):
        
        item_emb = trans_to_cuda(self.embedding.weight)
        text_emb = trans_to_cuda(self.text_embedding.weight)
        item_emb, text_emb = self.HyperGraph(self.adjacency, item_emb, text_emb)
        # obtain session emb
        sess_id_emb, sess_mo_emb, sess_id_pro_emb, sess_mo_pro_emb = self.generate_sess_emb(item_emb, text_emb, session_item, session_len, reversed_sess_item, mask)  # batch内session embeddings
        #  ID explicit loss
        item_co_loss = self.id_co_loss(item_emb, self.pos_item, self.neg_item, self.pos_weight)
        #  sequence proxy loss
        pro_loss = self.seq_pro_loss(sess_id_emb, sess_mo_emb, sess_id_pro_emb, sess_mo_pro_emb)
        # sequence criterion loss
        cri_loss = self.seq_cri_loss(item_emb, text_emb, sess_id_emb, sess_mo_emb,flag, target)

        con_loss = self.item_beta*item_co_loss + self.pro_beta*pro_loss + self.cri_beta*cri_loss


        return item_emb, text_emb, sess_id_emb, sess_mo_emb, con_loss


def perform(model, i, data):
    tar, flag, session_len, session_item, reversed_sess_item, mask = data.get_slice(i)
    session_item = trans_to_cuda(torch.Tensor(session_item).long())
    session_len = trans_to_cuda(torch.Tensor(session_len).long())
    tar = trans_to_cuda(torch.Tensor(tar).long())
    flag = trans_to_cuda(torch.Tensor(flag).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    reversed_sess_item = trans_to_cuda(torch.Tensor(reversed_sess_item).long())
    item_id_emb, item_mo_emb, sess_id_emb, sess_mo_emb, con_loss = model(session_item, flag, session_len, reversed_sess_item, mask, tar)
    scores_co = torch.mm(sess_id_emb, torch.transpose(item_id_emb, 1, 0))
    scores_mo = torch.mm(sess_mo_emb, torch.transpose(item_mo_emb, 1, 0))
    scores = scores_co + scores_mo
    scores = trans_to_cuda(scores)
    return tar, scores, con_loss

def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    torch.autograd.set_detect_anomaly(True)
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size) 
    for i in slices:
        model.zero_grad()
        targets, scores, con_loss = perform(model, i, train_data)
        loss = model.loss_function(scores + 1e-8, targets)
        loss = loss + con_loss
        loss.backward()
        #        print(loss.item())
        model.optimizer.step()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)
    top_K = [1, 5, 10, 20]
    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []
        metrics['ndcg%d' % K] = []
    print('start predicting: ', datetime.datetime.now())

    model.eval()
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        tar, scores, con_loss = perform(model, i, test_data)
        scores = trans_to_cpu(scores).detach().numpy()
        index = np.argsort(-scores, 1)
        tar = trans_to_cpu(tar).detach().numpy()
        for K in top_K:
            for prediction, target in zip(index[:, :K], tar):
                metrics['hit%d' % K].append(np.isin(target, prediction))
                if len(np.where(prediction == target)[0]) == 0:
                    metrics['mrr%d' % K].append(0)
                    metrics['ndcg%d' % K].append(0)
                else:
                    metrics['mrr%d' % K].append(1 / (np.where(prediction == target)[0][0] + 1))
                    metrics['ndcg%d' % K].append(1 / (np.log2(np.where(prediction == target)[0][0] + 2)))
    return metrics, total_loss



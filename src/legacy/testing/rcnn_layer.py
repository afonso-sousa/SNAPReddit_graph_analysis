import dgl.function as fn
import torch
import torch.nn as nn


class RGCNLayer(nn.Module):
    '''
    The abstract layer of a R-GCN
    '''

    def __init__(self, in_feat, out_feat, bias=None, activation=None,
                 self_loop=False, dropout=0.0):

        super(RGCNLayer, self).__init__()

        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        # Bias vector - Size: out_feat
        if self.bias == True:
            self.bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

        # Weight for self loop - Size: in_feat X out_feat
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        # Dropout according to the parameter
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    # Define how propagation is done in subclass
    def propagate(self, g):
        raise NotImplementedError

    def forward(self, g):
        # First of all I consider the self_loop contribution
        if self.self_loop:
            loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            if self.dropout is not None:
                loop_message = self.dropout(loop_message)

        # The propagation is implemented in each specific RGCN layer
        self.propagate(g)

        # Apply bias, loop message, and activation
        node_repr = g.ndata['h']
        if self.bias:
            node_repr = node_repr + self.bias
        if self.self_loop:
            node_repr = node_repr + loop_message
        if self.activation:
            node_repr = self.activation(node_repr)

        # Output of the layer applying the previous operations
        g.ndata['h'] = node_repr


class RGCNBasisLayer(RGCNLayer):
    """
    RCNBasisLayer implements the interface of the RGCLayer
    """

    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, is_input_layer=False):

            # For the basis layer, the self_loop and dropout have default values,
            # respectively False and 0.0
        super(RGCNBasisLayer, self).__init__(
            in_feat, out_feat, bias, activation)

        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.is_input_layer = is_input_layer

        # If the number of features is less than the number of relations, I apply
        # the regularization
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # Add bases weights
        # The weight is a 3-D tensor, because it consider input features, output
        # features (matrix) and all possible relationships related to input and
        # output features (third dimension)
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                self.out_feat))
        if self.num_bases < self.num_rels:
            # Linear combination coefficients
            # This is the regularization through basis decomposition
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels,
                                                    self.num_bases))

        nn.init.xavier_uniform_(
            self.weight, gain=nn.init.calculate_gain('relu'))

        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp,
                                    gain=nn.init.calculate_gain('relu'))

    def propagate(self, g):
        if self.num_bases < self.num_rels:
            # Generate all weights from bases
            weight = self.weight.view(self.num_bases,
                                      self.in_feat * self.out_feat)
            weight = torch.matmul(self.w_comp, weight).view(
                self.num_rels, self.in_feat, self.out_feat)
        else:
            weight = self.weight

        if self.is_input_layer:
            def msg_func(edges):
                # For input layer, matrix multiply can be converted to be
                # An embedding lookup using source node id
                embed = weight.view(-1, self.out_feat)
                index = edges.data['type'] * self.in_feat + edges.src['id']
                return {'msg': embed.index_select(0, index) * edges.data['norm']}
        else:
            def msg_func(edges):
                w = weight.index_select(0, edges.data['type'])
                msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
                msg = msg * edges.data['norm']
                return {'msg': msg}

        g.update_all(msg_func, fn.sum(msg='msg', out='h'), None)


class RGCNBlockLayer(RGCNLayer):
    # in_feat and out_feat correspond to the number of neurons for each block layer
    def __init__(self, in_feat, out_feat, num_rels, num_bases, bias=None,
                 activation=None, self_loop=False, dropout=0.0):

        super(RGCNBlockLayer, self).__init__(in_feat, out_feat, bias,
                                             activation, self_loop=self_loop,
                                             dropout=dropout)
        self.num_rels = num_rels
        self.num_bases = num_bases

        assert self.num_bases > 0

        self.out_feat = out_feat
        self.submat_in = in_feat // self.num_bases
        self.submat_out = out_feat // self.num_bases

        # Assuming in_feat and out_feat are both divisible by num_bases
        # Block diagonal regularization
        self.weight = nn.Parameter(torch.Tensor(
            self.num_rels, self.num_bases * self.submat_in * self.submat_out))
        nn.init.xavier_uniform_(
            self.weight, gain=nn.init.calculate_gain('relu'))

    def msg_func(self, edges):

        # edges.data[type] contains ids of sampled relations. Such ids are used
        # as indexes to get the rows from the weight matrix of the block layer
        # Before the reshape (.view()) the size of weights is
        # (len(edges), self.num_bases * self.submat_in * self.submat_out)
        # After the reshape we obtain a 3-dimensional tensor
        print('edges.data[type]')
        print(edges.data['type'])
        print(edges.data['type'].size())
        weight = self.weight.index_select(0, edges.data['type'])
        weight = weight.view(-1, self.submat_in, self.submat_out)

        # Take the embedding values of source nodes resulting from the sampling process
        node = edges.src['h'].view(-1, 1, self.submat_in)

        # Batch matrix matrix multiplication between the embedding values of nodes and the weights
        msg = torch.bmm(node, weight).view(-1, self.out_feat)

        return {'msg': msg}

    def propagate(self, g):
        # The embedding value of each node is update according to the message
        # and then is multiplied with the norm value
        g.update_all(self.msg_func, fn.sum(
            msg='msg', out='h'), self.apply_func)

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}


class BaseRGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases=-1,
                 num_hidden_layers=1, dropout=0, use_cuda=False):
        super(BaseRGCN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_cuda = use_cuda

        # create rgcn layers
        self.build_model()

        # create initial features
        self.features = self.create_features()

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    # initialize feature for each node
    def create_features(self):
        return None

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self, g):
        # The first step of the forward returns graph of the data
        # Data related to the nodes include ids (no features for nodes) and norms
        if self.features is not None:
            g.ndata['id'] = self.features
        for layer in self.layers:
            layer(g)
        return g.ndata.pop('h')

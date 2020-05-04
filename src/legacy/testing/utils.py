import numpy as np
import dgl

def get_adj_and_degrees(num_nodes, triplets):
    """ 
    Get adjacency list and degrees of the graph
    
    Arguments:
    
    num_nodes -- (int) number of nodes in the graph
    triplets -- (matrix) size(number of training samples, 3). Each row of the
                matrix is in the form: [subject_id, relation_id, object_id]
    
    Returns:
    
    adj_list -- (list) size(number of nodes, node degree)
    degrees -- (vector) size(, number of nodes). Each element of the vector is
               filled with the degree of the node.
    """
    adj_list = [[] for _ in range(num_nodes)]
    for i,triplet in enumerate(triplets):
        adj_list[triplet[0]].append([i, triplet[2]])
        adj_list[triplet[2]].append([i, triplet[0]])

    degrees = np.array([len(a) for a in adj_list])
    adj_list = [np.array(a) for a in adj_list]
    return adj_list, degrees

  
def sample_edge_neighborhood(adj_list, degrees, n_triplets, sample_size):
    """ 
    Edge neighborhood sampling to reduce graph size for training purposes
    
    Arguments:
    
    adj_list -- (structure) size(number of nodes, node degree)
    degrees -- (vector) size(, number of nodes). Each element of the vector is
               filled with the degree of the node
    n_triplets -- (int) number of triples in the training data
    sample_size -- (int) number of edges to sample in each iteration (parameter)
    
    Returns:
    
    edges -- list of edge indexes as result of the sampling process
    """

    edges = np.zeros((sample_size), dtype=np.int32)

    # Initialization for sampling of nodes
    sample_counts = np.array([d for d in degrees])
    picked = np.array([False for _ in range(n_triplets)])
    seen = np.array([False for _ in degrees])

    for i in range(0, sample_size):
        weights = sample_counts * seen

        if np.sum(weights) == 0:
            weights = np.ones_like(weights)
            weights[np.where(sample_counts == 0)] = 0

        probabilities = (weights) / np.sum(weights)
        
        # Choose vertex according to the computed probabilities
        # The number of chosen vertices is based on the sample size parameter
        chosen_vertex = np.random.choice(np.arange(degrees.shape[0]),
                                         p=probabilities)
        
        # Store all edges and nodes linked to the chosen vertex
        chosen_adj_list = adj_list[chosen_vertex]
        seen[chosen_vertex] = True
        
        # On the basis of the chosen vertex, randomly choose the edge and pick it up
        # from the adjacency list and store its value in the edge number
        chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
        chosen_edge = chosen_adj_list[chosen_edge]
        edge_number = chosen_edge[0]
        
        # If the edge has been already picked, 
        # choose another one according to the same principles
        while picked[edge_number]:
            chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
            chosen_edge = chosen_adj_list[chosen_edge]
            edge_number = chosen_edge[0]
        
        edges[i] = edge_number
        other_vertex = chosen_edge[1]
        picked[edge_number] = True
        sample_counts[chosen_vertex] -= 1
        sample_counts[other_vertex] -= 1
        seen[other_vertex] = True
        
        # Return all edges from the sampling process
    return edges

  
def generate_sampled_graph_and_labels(triplets, sample_size, split_size,
                                      num_rels, adj_list, degrees,
                                      negative_rate):
    """
    Get training graph and signals
    First perform edge neighborhood sampling on graph, then perform negative
    sampling to generate negative samples
    
    Arguments:
    
    triplets -- (matrix) size (num of triples in the training set, 3)
    sample_size -- (int) number of nodes to take as samples
    split_size -- (int) parameter to split the graph from (0 to 1)
    num_rels -- (int) number of unique relations in the training set
    adj_list -- (list) size(number of nodes, node degree)
    degrees -- (vector) size(, number of nodes).
    negative_rate -- (parameter)
    
    Returns:
    
    g -- DGL graph
    uniq_v -- (vector) unique nodes
    rel -- (vector) relations
    norm -- (vector) normalized degree values of each node
    samples -- (matrix) triple samples considering positive and negative samples
    labels -- (vector) size(,positive + negative samples). The value of each element
              is 1 for positive samples and 0 for negative samples.
    """
    
    # Perform edge neighbor sampling to take a subset of edges
    edges = sample_edge_neighborhood(adj_list, degrees, len(triplets),
                                     sample_size)

    # Relabel nodes to have consecutive node ids
    edges = triplets[edges] 
    src, rel, dst = edges.transpose()
    # Get unique vertices: such vertices are the result of the sampling. 
    # At the end of this process, we obtain an array of new triplets:
    # we use as node indexes the values of the array to reconstruct
    # the original src and dst
    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack((src, rel, dst)).transpose()

    # Negative sampling
    samples, labels = negative_sampling(relabeled_edges, len(uniq_v),
                                        negative_rate)

    # Further split graph, only half of the edges will be used as graph
    # structure, while the rest half is used as unseen positive samples
    # The graph is splitted according to split_size, picking up randomly
    # values between 0 and sample_size. 
    # According to these random values, I create the new src, dst, and rel arrays
    split_size = int(sample_size * split_size)
    graph_split_ids = np.random.choice(np.arange(sample_size),
                                       size=split_size, replace=False)
    src = src[graph_split_ids]
    dst = dst[graph_split_ids]
    rel = rel[graph_split_ids]
   
    # Build DGL graph
    # The number of nodes is equal to unique nodes after the sampling process
    # The number of edges is equal to src*2, because we consider in and out edges
    print("# sampled nodes: {}".format(len(uniq_v)))
    print("# sampled edges: {}".format(len(rel) * 2))
    g, rel, norm = build_graph_from_triplets(len(uniq_v), num_rels,
                                             (src, rel, dst))
    
    return g, uniq_v, rel, norm, samples, labels

  
def comp_deg_norm(g):
    """
    Apply a normalization to the values of input degree for each node
    
    Arguments:
    
    g -- DGL Graph
    
    Returns:
    
    norm -- np.array of normalized degree values - 1 X num of nodes
    """
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    # If the in_deg is 0, the result of normalization is inf (put 0 in this case)
    norm[np.isinf(norm)] = 0

    return norm

  
def build_graph_from_triplets(num_nodes, num_rels, triplets):
    """ 
    Create a DGL graph. The graph is bidirectional because RGCN authors
    use reversed relations. This function also generates edge type and
    normalization factor (reciprocal of node incoming degree)
    
    Arguments:
    
    num_nodes -- Number of nodes (result of the sampling)
    num_rels -- Number of relations (doubled compared to the original ones)
    triplets -- Three different arrays: src, rel, dst (result of the splitted graph)
    
    Returns:
    
    g -- High-level representation of the graph created using the DGL library
    rel -- (vector) vector of all relations
    norm -- (vector) normalized-degree of nodes
    """
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    src, rel, dst = triplets
    
    # The bidirectional graph is created (nodes are doubled)
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    
    # The relations are doubled adding the number of different relations 
    rel = np.concatenate((rel, rel + num_rels))
    
    # Create the edges array
    edges = sorted(zip(dst, src, rel))    
    dst, src, rel = np.array(edges).transpose() # 3 x Num of nodes
    g.add_edges(src, dst)
    
    norm = comp_deg_norm(g)
    print("# nodes: {}, # edges: {}".format(num_nodes, len(src)))
    
    return g, rel, norm

  
def build_test_graph(num_nodes, num_rels, edges):
    src, rel, dst = edges.transpose()
    print("Test graph:")
    return build_graph_from_triplets(num_nodes, num_rels, (src, rel, dst))

  
def negative_sampling(pos_samples, num_entity, negative_rate):
    """
    Apply negative sampling
    
    Arguments:
    
    pos_samples -- Relabeled edges according to the sampling process
    num_entity -- Number of entities
    negative_rate -- Negative rate parameter (default 10)
    
    Returns:
    
    np.array, labels -- np.array contains a concatenation of positive samples
                        and negative samples. 
                        labels is an array where elements are equal to 1 for 
                        positive samples and 0 for negative samples
    """
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate
    
    # Construct an array of negative samples by repeating 
    # the positive samples for the negative_rate value (np.tile).
    # Then, I create a label array to perform the logistic regression related to
    # the negative sampling. I initialize the first size_of_batch of labels to 1
    # and the other values are equal to 0.
    neg_samples = np.tile(pos_samples, (negative_rate, 1))        
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[: size_of_batch] = 1
    
    # Create negative samples replacing the subject or the object according
    # to the random probability generated in choices
    # 
    # If the positive samples are the following: 
    # [[1540  154 2254]
    # [ 510  193 1540]
    # [1540   55 1269]...]
    #
    # The negative samples will be the following:
    # [[1540  154 1750]
    #  [1051  193 1540]
    #  [85   55 1269]...]
    #
    # In other words we generate wrong triples
    values = np.random.randint(num_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]
    
    return np.concatenate((pos_samples, neg_samples)), labels

import torch
# Functions for Evaluation
def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices

  
def perturb_and_get_rank(embedding, w, a, r, b, num_entity, batch_size=100):
    """ 
    Perturb one element in the triplets
    
    """
    n_batch = (num_entity + batch_size - 1) // batch_size
    ranks = []
    for idx in range(n_batch):
        print("batch {} / {}".format(idx, n_batch))
        batch_start = idx * batch_size
        batch_end = min(num_entity, (idx + 1) * batch_size)
        batch_a = a[batch_start: batch_end]
        batch_r = r[batch_start: batch_end]
        emb_ar = embedding[batch_a] * w[batch_r]
        emb_ar = emb_ar.transpose(0, 1).unsqueeze(2) # size: D x E x 1
        emb_c = embedding.transpose(0, 1).unsqueeze(1) # size: D x 1 x V
        # out-prod and reduce sum
        out_prod = torch.bmm(emb_ar, emb_c) # size D x E x V
        score = torch.sum(out_prod, dim=0) # size E x V
        score = torch.sigmoid(score)
        target = b[batch_start: batch_end]
        ranks.append(sort_and_rank(score, target))
    return torch.cat(ranks)


# return MRR (raw), and Hits @ (1, 3, 10)
def evaluate(test_graph, model, test_triplets, num_entity, hits=[], eval_bz=100):
    with torch.no_grad():
        
        # During the evaluation we work with the entire graph
        
        # embedding and relations without gradients
        embedding, w = model.evaluate(test_graph)
        s = test_triplets[:, 0]
        r = test_triplets[:, 1]
        o = test_triplets[:, 2]

        # perturb subject
        ranks_s = perturb_and_get_rank(embedding, w, o, r, s, num_entity, eval_bz)
        # perturb object
        ranks_o = perturb_and_get_rank(embedding, w, s, r, o, num_entity, eval_bz)

        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1 # change to 1-indexed

        mrr = torch.mean(1.0 / ranks.float())
        print("MRR (raw): {:.6f}".format(mrr.item()))

        for hit in hits:
            avg_count = torch.mean((ranks <= hit).float())
            print("Hits (raw) @ {}: {:.6f}".format(hit, avg_count.item()))
    return mrr.item()
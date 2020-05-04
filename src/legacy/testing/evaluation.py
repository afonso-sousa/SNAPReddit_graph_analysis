import torch


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
        emb_ar = emb_ar.transpose(0, 1).unsqueeze(2)  # size: D x E x 1
        emb_c = embedding.transpose(0, 1).unsqueeze(1)  # size: D x 1 x V
        # out-prod and reduce sum
        out_prod = torch.bmm(emb_ar, emb_c)  # size D x E x V
        score = torch.sum(out_prod, dim=0)  # size E x V
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
        ranks_s = perturb_and_get_rank(
            embedding, w, o, r, s, num_entity, eval_bz)
        # perturb object
        ranks_o = perturb_and_get_rank(
            embedding, w, s, r, o, num_entity, eval_bz)

        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1  # change to 1-indexed

        mrr = torch.mean(1.0 / ranks.float())
        print("MRR (raw): {:.6f}".format(mrr.item()))

        for hit in hits:
            avg_count = torch.mean((ranks <= hit).float())
            print("Hits (raw) @ {}: {:.6f}".format(hit, avg_count.item()))
    return mrr.item()

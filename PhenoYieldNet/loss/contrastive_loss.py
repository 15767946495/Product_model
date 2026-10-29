import torch
from torch import nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size

        device = torch.device(device)
        self.register_buffer("temperature", torch.tensor(temperature).to(device))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float().to(device))

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)

        return loss
    
class MoCoLoss(nn.Module):
    def __init__(self, batch_size, device, temperature=0.07, momentum=0.999):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.momentum = momentum

        device = torch.device(device)
        self.register_buffer("queue", torch.zeros(batch_size, 128).to(device))  # Assuming 128 is the feature dimension
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long).to(device))

    def forward(self, query, key):
        """
        query: output from the query encoder
        key: output from the key encoder
        """
        # Normalize the embeddings
        query = F.normalize(query, dim=1)
        key = F.normalize(key, dim=1)

        # Concatenate the query and key
        representations = torch.cat([query, key], dim=0)

        # Compute similarity matrix
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        # Get positive and negative similarities
        pos_sim = torch.diag(similarity_matrix, self.batch_size)
        neg_sim = similarity_matrix[~torch.eye(similarity_matrix.size(0), dtype=bool)].view(similarity_matrix.size(0), -1)

        # Compute the loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1) / self.temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)

        loss = F.cross_entropy(logits, labels)

        # Update the queue
        self._update_queue(key)

        return loss

    def _update_queue(self, key):
        batch_size = key.size(0)
        ptr = int(self.queue_ptr)

        # Update the queue with the new keys
        self.queue[ptr:ptr + batch_size] = key.detach()
        ptr = (ptr + batch_size) % self.batch_size
        self.queue_ptr[0] = ptr



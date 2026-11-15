import torch
from sklearn.cluster import KMeans
import torch.nn as nn

def generate_pseudo_labels(encoded_views, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(encoded_views)
    pseudo_labels = kmeans.labels_
    return pseudo_labels

class ClusteringLayer(nn.Module):

    def __init__(self, n_clusters, input_dim, alpha=1.0, weights=None):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.clusters = nn.Parameter(torch.randn(n_clusters, input_dim))
        if weights is not None:
            self.clusters.data = torch.tensor(weights, dtype=torch.float32)

    def forward(self, inputs):
        dist_squared = torch.sum((inputs.unsqueeze(1) - self.clusters) ** 2, 2)
        q = 1.0 / (1.0 + dist_squared / self.alpha)
        q **= (self.alpha + 1.0) / 2.0
        q /= torch.sum(q, dim=1, keepdim=True)
        return q

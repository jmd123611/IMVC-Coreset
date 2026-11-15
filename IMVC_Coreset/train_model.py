import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.cluster import KMeans
from cluster import ClusteringLayer
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(encoders, decoders, data_loader, class_num, real_label, num_epochs=200):
    loss_function = nn.MSELoss()
    optimizers = [optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.005) for encoder, decoder in zip(encoders, decoders)]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for epoch in range(num_epochs):
        encoded_views_per_epoch_per_view = [[] for _ in encoders]
        for views_batch in data_loader:
            losses = []
            encoded_views = []
            for i, (encoder, decoder, view) in enumerate(zip(encoders, decoders, views_batch)):
                view = view.float().to(device)
                encoded = encoder(view)
                encoded_views.append(encoded)
                encoded_views_per_epoch_per_view[i].append(encoded.detach())
                decoded = decoder(encoded)
                loss = loss_function(decoded, view)
                losses.append(loss)
                optimizer = optimizers[i]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss = sum(losses)
        concatenated_views_per_view = [torch.cat(views, dim=0) for views in encoded_views_per_epoch_per_view]
        concatenated_encoded_views = torch.cat(concatenated_views_per_view, dim=1).detach().cpu().numpy()
    return (encoders, decoders, concatenated_encoded_views)

def train_model_with_new_loss(encoders, decoders, data_loader, num_epochs, class_num, con_labels):
    con_labels = torch.tensor(con_labels, dtype=torch.long).to(device)
    concatenated_encoded_views = None
    mse_loss_function = nn.MSELoss()
    ce_loss_function = nn.CrossEntropyLoss()
    encoded_views_per_encoder = [[] for _ in encoders]
    for views_batch in data_loader:
        for i, (encoder, _, view) in enumerate(zip(encoders, decoders, views_batch)):
            view = view.float().to(device)
            encoded = encoder(view)
            encoded_views_per_encoder[i].append(encoded.detach().cpu().numpy())
    clustering_layers = []
    for i, encoded_views in enumerate(encoded_views_per_encoder):
        encoded_views = np.vstack(encoded_views)
        kmeans = KMeans(n_clusters=class_num)
        kmeans.fit(encoded_views)
        initial_cluster_centers = kmeans.cluster_centers_
        clustering_layer = ClusteringLayer(n_clusters=class_num, input_dim=class_num, weights=initial_cluster_centers)
        clustering_layer.to(device)
        clustering_layers.append(clustering_layer)
    optimizers = [optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(clustering_layer.parameters()), lr=0.001) for encoder, decoder, clustering_layer in zip(encoders, decoders, clustering_layers)]
    for epoch in range(num_epochs):
        encoded_views_per_epoch_per_view = [[] for _ in encoders]
        for views_batch in data_loader:
            losses = []
            for i, (encoder, decoder, view) in enumerate(zip(encoders, decoders, views_batch)):
                optimizers[i].zero_grad()
                view = view.float().to(device)
                encoded = encoder(view)
                encoded_views_per_epoch_per_view[i].append(encoded.detach())
                decoded = decoder(encoded)
                recon_loss = mse_loss_function(decoded, view)
                soft_labels = clustering_layers[i](encoded.to(device))
                entropy_loss = ce_loss_function(soft_labels, con_labels)
                total_loss = recon_loss + 0.01 * entropy_loss
                total_loss.backward()
                optimizers[i].step()
                losses.append(total_loss.item())
            avg_loss = sum(losses) / len(losses)
        if epoch == num_epochs - 1:
            concatenated_views_per_view = [torch.cat(views, dim=0) for views in encoded_views_per_epoch_per_view]
            concatenated_encoded_views = torch.cat(concatenated_views_per_view, dim=1).detach().cpu().numpy()
    return (encoders, decoders, concatenated_encoded_views)

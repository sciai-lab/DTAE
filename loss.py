import numpy as np
import matplotlib.pyplot as plt
import higra as hg
import torch
from tqdm import tqdm

#=========================================
#=            Helper Functions           =
#=========================================

def get_centroids(X, high_dim_clusters,K,device="cpu",dim=2):
    index_sets = [np.argwhere(high_dim_clusters==i) for i in np.arange(K)]
    centers = torch.zeros((K,dim),device=device)
    for i,indices in enumerate(index_sets):
        center = torch.mean(X[indices,:],axis=0)
        centers[i] = center
    return centers

def get_closest_2_centroids(X,centroids,device="cpu"):
    closest =torch.zeros((X.shape[0],2),dtype=torch.long,device=device)
    for i,point in enumerate(X) :
        dist = torch.sum((point - centroids)**2,axis=1)
        indices = torch.topk(-dist,2)[1]
        if indices[1] < indices[0]:
            indices = torch.flip(indices,[0])
        closests = centroids[indices]
        closest[i] = indices
    return closest

def cdist_custom(X,Y,eps=1e-6,device="cpu"):
    epsilon = torch.FloatTensor([eps]).to(device)
    distances = X.unsqueeze(1) - Y
    distances = distances**2
    distances = torch.sum(distances,axis=2)
    distances = torch.max(epsilon,distances) # eps + distances ?
    distances = torch.sqrt(distances)
    
    return distances

def get_ghost_centroids_low_dim(centroids, end_vertices, neighbours,ratio=0.1):
    return centroids[end_vertices] + (centroids[end_vertices] - centroids[neighbours])*ratio

#=========================================
#=             Loss Functions            =
#=========================================

def cosine_loss(triplets, centroids):
    x1 = centroids[triplets[:,1]] - centroids[triplets[:,0]]
    x2 = centroids[triplets[:,2]] - centroids[triplets[:,1]]
    cos_sim = torch.nn.CosineSimilarity(dim = 1, eps= 1e-08)
    loss = cos_sim(x1,x2)
    return 2-(loss+1) # Rescale between 0 and 2 and to get cosine distance

def push_pull_crispness(X,centers,closests_2,x_labels,geo_dist,eps=1e-6,device="cpu"):

    #Fix for sqrt issues in pytorch cdist
    distances = cdist_custom(centers,X,eps=eps,device=device)
    distances_centroids = cdist_custom(centers,centers,eps=eps,device=device)

    # Geodesic distance to the two closest centroids
    geodesic_distances = geo_dist[closests_2.T[0],closests_2.T[1]]
    idx = torch.arange(X.shape[0])
    # Normalize distances
    norm = distances_centroids[x_labels.T[0],x_labels.T[1]]
    
    distances_closests = ((distances[closests_2[:,0],idx]+distances[closests_2[:,1],idx])/norm)**2  
    distances_truth = ((distances[x_labels[:,0],idx]+distances[x_labels[:,1],idx])/norm)**2

    # > 0 means that we are wrong and =0 that we have the right clustering
    # We break the relationship if factor > 1
    return geodesic_distances*(-distances_closests + distances_truth), distances_truth

def finetune_loss(X, groundtruth,device="cpu",real_cosine_dist=False,ghost_centroids_distance_ratio=0.5):
    centroids  = get_centroids(X,groundtruth.high_dim_assignments,groundtruth.K,device=groundtruth.device)

    ghost_centroids = get_ghost_centroids_low_dim(centroids, groundtruth.end_vertices, groundtruth.neighbours,ratio=ghost_centroids_distance_ratio)
    centroids = torch.cat([centroids,ghost_centroids], dim=0)
    new_closests_centroids = get_closest_2_centroids(X,centroids,device=groundtruth.device)
    
    loss_push_pull, loss_crispness = push_pull_crispness(X,centroids,new_closests_centroids,groundtruth.closests_centroids,groundtruth.geo_dist,device=groundtruth.device)

    loss_cos = cosine_loss(groundtruth.triplets, centroids)
    if real_cosine_dist == True:
        mse = torch.nn.MSELoss(reduction="none")
        loss_cos = mse(loss_cos,groundtruth.cos_dist)

    return torch.mean(loss_push_pull),torch.mean(loss_crispness), torch.mean(loss_cos)







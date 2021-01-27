import torch
import numpy as np
import higra as hg
from sklearn.cluster import KMeans
import queue
import time

class Groundtruth:
    def __init__(self,X_high_dim,n_jobs=8,K=50,device="cpu"):
        X = X_high_dim.detach().numpy()
        
        start = time.time()
        print("Computing k-means on high dim data ....")
        kmeans = KMeans(n_clusters=K, random_state=0,n_jobs=n_jobs).fit(X)
        high_dim_assignments = kmeans.labels_
        centroids = np.array(kmeans.cluster_centers_)

        print("Computing the densities ....")
        closests_centroids, centroid_pairs, densities = densities_hebbian_learning(X,centroids)

        print("Building the minimum spanning tree ....")
        adjacency_matrix = adjacency_matrix_from_densities(K,centroid_pairs,densities)
        graph, edge_weights = hg.adjacency_matrix_2_undirected_graph(adjacency_matrix)

        tree, altitudes_ = hg.bpt_canonical(graph,edge_weights)
        mst = hg.get_attribute(tree, "mst")

        print("Computing ghost centroids ....")
        end_vertices, neighbours, ghosts = get_ghost_centroids(mst,centroids)
        
        centroids = np.concatenate([centroids,ghosts])
        
        print("Recomputing closest centroids ....")
        new_closests_centroids, _,_ = densities_hebbian_learning(X,centroids)
        

        # Only update ones part of the alst cluster of a branch
        mask = np.zeros_like(high_dim_assignments,dtype=np.bool)
        for v in end_vertices:
            mask[high_dim_assignments == v] = 1

        closests_centroids[mask] = new_closests_centroids[mask]

        mst = add_ghost_edges(mst,end_vertices,ghosts)
        
        print("Computing geodesic distances ....")
        
        geo_dist = mst_geodesic_distances(mst)
        geo_dist = geo_dist 
     
        print("Finding triplets for cosine loss ....")
        triplets = get_triplets(mst,K)

        print("Computing cosine distance ....")
        cos_dist = cosine_distance(triplets, torch.FloatTensor(centroids))

        
        self.K = K
        self.device = device

        self.closests_centroids = closests_centroids
        self.high_dim_assignments = high_dim_assignments
        self.end_vertices = end_vertices
        self.neighbours = neighbours
        self.geo_dist = torch.FloatTensor(geo_dist).to(device)
        self.triplets = triplets.to(device)
        self.cos_dist = cos_dist.to(device)
        print("Done with initializaion, took {} seconds".format(time.time()-start))

#=========================================
#=            Helper functions           =
#=========================================

def densities_hebbian_learning(X, centers):
    closest = np.zeros(shape=(X.shape[0],2),dtype=np.int)
    for i,point in enumerate(X) :
        dist = (np.sum((point - centers)**2,axis=1))
        closests = np.argpartition(dist,2)[:2]
        if closests[1] < closests[0] :
            closest[i] = closests[::-1]
        else :
            closest[i] = closests
    
    pairs,densities = np.unique(closest,axis=0,return_counts=True)
    return closest, pairs, densities

def adjacency_matrix_from_densities(K, centroid_pairs, densities):
    adjacency_matrix = np.zeros(shape=(K,K)) + 2 # Issue if graph is not connected
    np.fill_diagonal(adjacency_matrix, 0)

    for i,pair in enumerate(centroid_pairs):
        dist = 1/densities[i] #To use the minmum spanning tree and not maximum
        adjacency_matrix[tuple(pair)] = dist
        adjacency_matrix[tuple(pair[::-1])] = dist

    return adjacency_matrix

def get_points_contributing_MST(mst,closests_centroids):
    present_pairs = np.array(mst.edge_list()).T
    points_contributing = np.zeros(shape=(closests_centroids.shape[0]))
    for pair in present_pairs:
        points_contributing[np.all(closests_centroids==pair,axis=1)] = 1
    return torch.BoolTensor(points_contributing)

def get_edge_weights(closests_centroids, centroid_pairs, densities):
    weights = np.zeros((closests_centroids.shape[0],))
    for pair,density in zip(centroid_pairs,densities):
        weights[np.all(closests_centroids==pair,axis=1)] = density
    print(weights)
    return torch.FloatTensor(weights)

def get_ghost_centroids(mst,centroids,ratio=0.1):
    end_vertices = np.where(mst.degree(np.arange(mst.num_vertices())) == 1)[0]

    neighbours = []
    for v in end_vertices :
        for neighbour in mst.adjacent_vertices(v):
            neighbours.append(neighbour)

    ghosts = centroids[end_vertices] + (centroids[end_vertices] - centroids[neighbours])*ratio
    
    return end_vertices, neighbours, ghosts

def mst_geodesic_distances(mst):
    # O(K^2)
    geo_distances = np.zeros(shape=(mst.num_vertices(),mst.num_vertices()))
    for root in range(mst.num_vertices()):
        already_seen = np.zeros(shape=(mst.num_vertices(),))
        distances = np.zeros(shape=(mst.num_vertices(),))

        already_seen[root] = 1
        Q = queue.Queue()
        Q.put(root)
        while not Q.empty():
            v = Q.get()
            for w in mst.adjacent_vertices(v):
                if not already_seen[w]:
                    already_seen[w] = 1
                    distances[w] = distances[v] + 1 # Edge weight = 1
                    Q.put(w)
        geo_distances[root] = distances
    
    return geo_distances

def add_ghost_edges(mst,end_vertices,ghosts):
    for i in end_vertices:
        v = mst.add_vertex()
        mst.add_edge(i,v)
    return mst

def get_triplets(mst,K):
    triplets = []
    for i in range(K):
        if mst.degree(i) == 2 :
            adjacent_vertices = []
            for v in mst.adjacent_vertices(i):
                adjacent_vertices.append(v)
            triplet = [adjacent_vertices[0],i,adjacent_vertices[1]]
            triplets.append(triplet)
    return torch.LongTensor(triplets)

def cosine_distance(triplets, centroids):
    x1 = centroids[triplets[:,1]] - centroids[triplets[:,0]]
    x2 = centroids[triplets[:,2]] - centroids[triplets[:,1]]
    cos_sim = torch.nn.CosineSimilarity(dim = 1, eps= 1e-08)
    loss = cos_sim(x1,x2)
    return 2-(loss+1) # Rescale between 0 and 2 and to get cosine distance



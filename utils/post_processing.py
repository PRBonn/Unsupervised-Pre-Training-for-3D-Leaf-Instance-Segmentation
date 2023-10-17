import numpy as np
import matplotlib.cm
from sklearn.cluster import AgglomerativeClustering
import open3d as o3d
import networkx as nx

def post_processing_best(points, embeddings, n_steps=4.):
    # points: N x 3 array with positions of the points 
    # embeddings: N x embedding_dim array with embeddings for each of the N points

    # estimate center of the plant
    center = points.mean(0)
    # parameter for the cutting radius depends on the max variance of points along x, y coords
    cutting_radius_init = min(np.max(np.abs(points - center),0)[:2]) # use this to define the radius to cut 
    cutting_radius_init *= 0.4 # save the tip of the smallest leaf

    # we want to cut out a cylinder with base centered at center_x, center_y, 0, radius: f(variance), height: inf
    # point P is inside of cylinder if d(P, center) < radius where d is the euclidean distance only on x | y coords
    step_size = cutting_radius_init/n_steps
    radius_to_cut = np.arange(cutting_radius_init, -step_size, -step_size) # last element is cutting zero, we cluster verything!

    # array to save instance id for each point
    instances = np.zeros((points.shape[0],))

    # for loop for cutting always less points, to expand the labels from outer regions
    for radius in radius_to_cut:

        idxs_of_points_to_cluster = ( np.sqrt( ((points[:,:2] - center[:2])**2).sum(1) ) > radius ) & ~instances.astype('bool')
        vec_to_cluster = embeddings[idxs_of_points_to_cluster,:]

        if vec_to_cluster.shape[0] == 0:
            return instances
        cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=0.8).fit(vec_to_cluster)
        n_clusters = cluster.n_clusters_

        if instances.sum() == 0: # no instances, first round
            instances[ idxs_of_points_to_cluster ] = cluster.labels_ + 1 # clusters id starts from 0 

            # compute centers of "leaves" 
            instances_centers = np.zeros((n_clusters, embeddings.shape[1]))
            for c in range(n_clusters):
                points_assigned = embeddings[ (instances == c+1) ]
                instances_centers[c,:] = points_assigned.mean(0)

        else: # need to assign to new or old ones

            assign = np.zeros((n_clusters,))
            for c in range(n_clusters): # iterating over the new clusters
                points_assigned =  embeddings[idxs_of_points_to_cluster][ cluster.labels_ == c ]
                new_center = points_assigned.mean(0)
                value = np.min(np.linalg.norm(new_center - instances_centers, axis=1))
                if value < 0.8: # threshold of clustering, below means that they can be assigned
                    assign[c] = np.argmin(np.linalg.norm(new_center - instances_centers, axis=1))
                else:
                    assign[c] = np.max(instances) + 1
            instances[ idxs_of_points_to_cluster ] = assign[cluster.labels_] + 1 # still labels with indexes from 0 
    return instances

def post_processing_graph(points, embeddings, graph_matrix, affinity="cosine", linkage="single"):
    # points: N x 3 array with positions of the points 
    # embeddings: N x embedding_dim array with embeddings for each of the N points

    # estimate center of the plant
    center = points.mean(0)
    # parameter for the cutting radius depends on the max variance of points along x, y coords
    radius = min(np.max(np.abs(points - center),0)[:2]) # use this to define the radius to cut 
    radius *= 0.4 # save the tip of the smallest leaf

    # array to save instance id for each point
    instances = np.zeros((points.shape[0],))

    # for loop for cutting always less points, to expand the labels from outer regions
    idxs_of_points_to_cluster = ( np.sqrt( ((points[:,:2] - center[:2])**2).sum(1) ) > radius )
    vec_to_cluster = embeddings[idxs_of_points_to_cluster,:]

    cluster = AgglomerativeClustering(n_clusters=None,  distance_threshold=0.8).fit(vec_to_cluster)
    n_clusters = cluster.n_clusters_

    instances[ idxs_of_points_to_cluster ] = cluster.labels_ + 1 # clusters id starts from 0 

    assignments = np.zeros((points.shape[0], n_clusters))
    ## attach one cluster at the source and the others to sink, loop ovber all of them and cut the graph 
    for c in range(n_clusters):
        src_probs = np.ones((len(points),))*1e-20
        snk_probs = np.ones((len(points),))*1e-20

        G = nx.Graph()
        G.add_node("source")
        G.add_node("sink")

        sources = np.where(instances == c+1)[0]
        sinks = np.where(instances == c+2)[0]

        if len(sinks) == 0: # this is only when we are at the last instance
            sinks = np.where(instances == c -1)[0] # get last cluster 

        src_probs[sources] = 1.
        snk_probs[sinks] = 1.

        for i in range(len(points)):
            G.add_node(i)
            G.add_edge("source",i)
            G["source"][i]["capacity"] = - 0.1 * np.log(src_probs[i])

            G.add_edge(i,"sink")
            G[i]["sink"]["capacity"] = - 0.1 *  np.log(snk_probs[i])

            neigh = np.where(graph_matrix[i])[0]
            for k in neigh:
                G.add_edge(i,k)
                G[i][k]["capacity"] = 10 * np.exp(1/(((embeddings[i].T @ embeddings[k]) / (1e-5 + np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[k]))) / 2))
        
        _ , partition = nx.minimum_cut(G, "source", "sink")
        reachable, non_reachable = partition

        if reachable.__len__() > non_reachable.__len__():
            non_reachable.remove("sink")
            assignments[ list(non_reachable), c ] = 1.
        else:
            reachable.remove("source")
            assignments[ list(reachable), c ] = 1.

    sum_over_assign = assignments.sum(1) # unique assignments
    sum_over_assign = sum_over_assign == 1
    # assigning everything that is not already assigned 
    instances[sum_over_assign*(~instances.astype(np.bool))] = np.where(assignments[sum_over_assign*(~instances.astype(np.bool))] == 1)[1] +1 # cluster 0 is cluster 1

    centers = np.zeros((n_clusters, embeddings.shape[1]))
    for inst in range(n_clusters):
        try:
            centers[inst] = embeddings[ instances == inst +1 ].mean(0)
        except:
            continue

    elements = np.where(instances == 0)[0]
    for item in elements:

        max_comp = 0
        assign = -1
        for cc in range(centers.shape[0]):
            value = (embeddings[item].T @ centers[cc]) / (1e-5 + np.linalg.norm(embeddings[item]) * np.linalg.norm(centers[cc]))
            if value >= max_comp:
                max_comp = value
                assign = cc

        instances[item] = assign + 1

    return instances

def post_processing_base(points, embeddings, n_steps=4., affinity="cosine", linkage="single"):
    from sklearn.cluster import AgglomerativeClustering
    # points: N x 3 array with positions of the points 
    # embeddings: N x embedding_dim array with embeddings for each of the N points

    # estimate center of the plant
    center = points.mean(0)
    # parameter for the cutting radius depends on the max variance of points along x, y coords
    cutting_radius_init = min(np.max(np.abs(points - center),0)[:2]) # use this to define the radius to cut 
    cutting_radius_init *= 0.4 # save the tip of the smallest leaf

    # we want to cut out a cylinder with base centered at center_x, center_y, 0, radius: f(variance), height: inf
    # point P is inside of cylinder if d(P, center) < radius where d is the euclidean distance only on x | y coords
    step_size = cutting_radius_init/n_steps
    radius_to_cut = np.arange(cutting_radius_init, -step_size, -step_size) # last element is cutting zero, we cluster verything!

    # array to save instance id for each point
    instances = np.zeros((points.shape[0],))
    # initialize n_cluster
    n_clusters = 0

    # for loop for cutting always less points, to expand the labels from outer regions
    for radius in radius_to_cut:

        # run clustering on embeddings
        if n_clusters == 0: # first time running clustering
            # takes the indexes of points not cut out
            idxs_of_points_to_cluster = ( np.sqrt( ((points[:,:2] - center[:2])**2).sum(1) ) > radius )
            # takes the embeddings of remaining points
            vec_to_cluster = embeddings[idxs_of_points_to_cluster,:]
            cluster = AgglomerativeClustering(n_clusters=None, affinity=affinity, linkage=linkage, distance_threshold=0.2).fit(vec_to_cluster)
            # save the number of potential leaves, use it for future clusterings
            n_clusters = cluster.n_clusters_
            instances[ idxs_of_points_to_cluster ] = cluster.labels_ + 1 # clusters id starts from 0 

            # compute centers of "leaves" 
            instances_centers = np.zeros((n_clusters,embeddings.shape[1]))
            for c in range(n_clusters):
                points_assigned = embeddings[ (instances == c+1) ]
                instances_centers[c,:] = points_assigned.mean(0)

            #compute_leaf_colors(points, instances)
        else: # we alreay have some instances fgrom outer regions
            # we remove points in the cylinder and assigned points
            idxs_of_points_to_cluster = ( np.sqrt( ((points[:,:2] - center[:2])**2).sum(1) ) > radius ) & ~instances.astype('bool')
            vec_to_cluster = embeddings[idxs_of_points_to_cluster,:]
            cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, linkage=linkage).fit(vec_to_cluster)

            assign = np.zeros((n_clusters,))
            for c in range(n_clusters):
                points_assigned = embeddings[idxs_of_points_to_cluster][ cluster.labels_ == c ]
                new_center = points_assigned.mean(0)
                assign[c] = np.argmin(np.linalg.norm(new_center - instances_centers, axis=1))

            instances[ idxs_of_points_to_cluster ] = assign[cluster.labels_] + 1 # still labels with indexes from 0 
    return instances

def compute_leaf_colors(points, leaf_ids):
    #leaf_ids = np.unique(leaf_ids, return_inverse=True)[1]
    leaf_list = np.unique(leaf_ids)
    cmap_leaves = matplotlib.cm.get_cmap("tab20")
    point_colors = np.zeros_like(points)

    for idx in leaf_list:
        if idx == 0:
            color = (0.0, 0.0, 0.0)
        else:
            color = cmap_leaves(idx/len(leaf_list))[:3]
        leaf_mask = leaf_ids == idx
        point_colors[leaf_mask] = np.asarray(color)

    # Create an o3d pointcloud
    leaf_seg_pcl = o3d.geometry.PointCloud()
    leaf_seg_pcl.points = o3d.utility.Vector3dVector(points)
    leaf_seg_pcl.colors = o3d.utility.Vector3dVector(point_colors)
    o3d.visualization.draw_geometries([leaf_seg_pcl])

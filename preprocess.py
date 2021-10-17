# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 21:27:41 2021

@author: rjafa
"""

import os
import numpy as np
import igraph
import itertools
import random
import logging

DIGG_BASE_DIR="D:/payan name\dataset/DeepInf/digg/"
vote_file=os.path.join(DIGG_BASE_DIR, "digg_votes1.csv")
graph_file=os.path.join(DIGG_BASE_DIR, "digg_friends.csv")
output = os.path.join(DIGG_BASE_DIR,
                "digg_processed/rw_sample_inf_100_1k_degree_3_31_ego_50_neg_1_restart_20")

min_active_neighbor = 3
min_inf = 100
max_inf = 1500
min_degree = 3
max_degree = 31
ego_size = 49
argsNegative = 1
restart_prob = 0.2
walk_length = 1000

logger = logging.getLogger(__name__)


def random_walk_with_restart(g, start, restart_prob):
    current = random.choice(start)
    stop = False
    while not stop:
        stop = yield current
        current = random.choice(start) if random.random() < restart_prob or g.degree(current)==0 \
                else random.choice(g.neighbors(current))
                
class Digg:
    v2id = {}               #{userId:convertedUserId,...}
    diffusion = {}          #{storyId:[(userId,timeStamp),...],...}
    adj_matrices = []
    features = []
    vertices = []
    labels = []
    structural_features = []
    
    
        
    def get_vid(self, u, readonly=False):
        if u not in self.v2id:
            if readonly:
                return -1
            newid = len(self.v2id)
            self.v2id[u] = newid
        return self.v2id[u]
    
    #create diffusion(which users vote a story)
    def load_vote(self):
        print("Load digg vote ...")
        with open(vote_file, "r") as f:
            for line in f:
                content = line.strip().split(',')
                t, u, vote = [int(x[1:-1]) for x in content]
                u = self.get_vid(u, readonly=True)
                if u > -1:
                    if vote not in self.diffusion:
                        self.diffusion[vote] = []
                    self.diffusion[vote].append((u, t))
            
    
    #create edge list [(convertedUserId, convertedUserId),...] from digg_friends.csv 
    def extract_edge_list(self, graph_file):
        edgelist = []
        print("Extrack edges from %s", graph_file)
        with open(graph_file, "r") as f:
            nu = 0
            
            for line in f:
                nu += 1
                if (nu+1) % 100000 == 0:
                    print("%d user profile proccessed" % (nu+1))
                    
                
                content = line.strip().split(',')
                u, v = int(content[-2][1:-1]), int(content[-1][1:-1])
                u, v = self.get_vid(u, readonly=False), self.get_vid(v, readonly=False)
                edgelist.append((u,v))
    
        return edgelist
    
    
    #create graph from edge list with igraph
    def load_graph(self):
        edgelist = self.extract_edge_list(graph_file)
        n_vertices = len(self.v2id)
        self.graph = igraph.Graph(len(self.v2id), directed=False)
        self.graph.add_edges(edgelist)
        self.graph.to_undirected()
        self.graph.simplify(multiple=True, loops=True)
        edgelist_path = os.path.join(output, "graph.edgelist")
        with open(edgelist_path, "w") as f:
            self.graph.write(f, format="edgelist")
        # add some fake vertices
        self.graph.add_vertices(ego_size)
        self.degree = self.graph.degree()
        
    def sort_diffusion(self):
        for k in self.diffusion:
            self.diffusion[k] = sorted(self.diffusion[k],
                                        key=lambda item: item[1])
    def summarize_diffusion(self):
        #array of votes count
        self.diffusion_size = [len(v) for v in self.diffusion.values()]
        print("mean diffusion length %.2f", np.mean(self.diffusion_size))
        print("max diffusion length %.2f", np.max(self.diffusion_size))
        print("min diffusion length %.2f", np.min(self.diffusion_size))
        for i in range(1, 10):
            print("%d-th percentile of diffusion length %.2f", i*10,
                    np.percentile(self.diffusion_size, i*10))
        print("95-th percentile of diffusion length %.2f",
                np.percentile(self.diffusion_size, 95))

    def summarize_graph(self):
        print("mean degree %.2f", np.mean(self.degree))
        print("max degree %.2f", np.max(self.degree))
        print("min degree %.2f", np.min(self.degree))
        for i in range(1, 10):
            print("%d-th percentile of degree %.2f", i*10,
                    np.percentile(self.degree, i*10))
        print("95-th percentile of degree %.2f",
                    np.percentile(self.degree, 95))
        
    def compute_structural_features(self):
        print("Computing rarity (reciprocal of degree)")
        degree = np.array(self.graph.degree())
        degree[degree==0] = 1
        rarity = 1. / degree
        print("Computing clustering coefficient..")
        cc = self.graph.transitivity_local_undirected(mode="zero")
        print("Computing pagerank...")
        pagerank = self.graph.pagerank(directed=False)
        print("Computing constraint...")
        """
        constraint = self.graph.constraint()
        logger.info("Computing closeness...")
        closeness = self.graph.closeness(cutoff=3)
        logger.info("Computing betweenness...")
        betweenness = self.graph.betweenness(cutoff=3, directed=False)
        logger.info("Computing authority_score...")
        """
        authority_score = self.graph.authority_score()
        print("Computing hub_score...")
        hub_score = self.graph.hub_score()
        print("Computing evcent...")
        evcent = self.graph.evcent(directed=False)
        print("Computing coreness...")
        coreness = self.graph.coreness()
        print("Structural feature computation done!")
        self.structural_features = np.column_stack(
                (rarity, cc, pagerank,
                 #constraint, closeness, betweenness,
                 authority_score, hub_score, evcent, coreness))

        with open(os.path.join(output, "vertex_feature.npy"), "wb") as f:
            np.save(f, self.structural_features)
         
            
    def create(self, u, p, t, label, user_affected_now):
        graph = self.graph

        active_neighbor, inactive_neighbor = [], []

        neighbors_u = graph.neighbors(u)
        for v in neighbors_u:
            if v in user_affected_now:
                active_neighbor.append(v)
            else:
                inactive_neighbor.append(v)
        if len(active_neighbor) < min_active_neighbor:
            return

        n = ego_size + 1
        n_active = 0
        ego = []
        if len(active_neighbor) < ego_size:
            # we should sample some inactive neighbors
            n_active = len(active_neighbor)
            ego = set(active_neighbor)
            start_for_randomWalk = active_neighbor + [u,]
            randomWalk = random_walk_with_restart(graph,
                start=start_for_randomWalk, restart_prob=restart_prob)
            
            #create ego with randomWalk
            for v in itertools.islice(randomWalk, walk_length):
                if v!=u and v not in ego:
                    ego.add(v)
                    if len(ego) == ego_size:
                        break
            ego = list(ego)
            if len(ego) < ego_size:
                return
                n_fake = ego_size - len(ego)
                print("generate %d fake vertices", n_fake)
                ego += list(range(self.n_vertices, self.n_vertices+n_fake))
        else:
            n_active = ego_size
            samples = np.random.choice(active_neighbor,
                    size=ego_size,
                    replace=False)
            ego += samples.tolist()
        ego.append(u)

        order = np.argsort(ego)
        ranks = np.argsort(order)

        subgraph = graph.subgraph(ego, implementation="create_from_scratch")
        adjacency = np.array(subgraph.get_adjacency().data)
        adjacency = adjacency[ranks][:, ranks]
        self.adj_matrices.append(adjacency)

        feature = np.zeros((n,2))
        for idx, v in enumerate(ego[:-1]):
            if v in user_affected_now:
                feature[idx, 0] = 1
        feature[n-1, 1] = 1
        self.features.append(feature)
        self.vertices.append(np.array(ego, dtype=int))
        self.labels.append(label)

        circle = subgraph.subgraph(ranks[:n_active], implementation="create_from_scratch")

        if len(self.labels) % 1000 == 0:
            print("Collected %d instances", len(self.labels))

    def dump_data(self):
        self.adj_matrices = np.array(self.adj_matrices)
        self.features = np.array(self.features)
        self.vertices = np.array(self.vertices)
        self.labels = np.array(self.labels)

        output_dir = output
        with open(os.path.join(output_dir, "adjacency_matrix.npy"), "wb") as f:
            np.save(f, self.adj_matrices)
        with open(os.path.join(output_dir, "influence_feature.npy"), "wb") as f:
            np.save(f, self.features)
        with open(os.path.join(output_dir, "vertex_id.npy"), "wb") as f:
            np.save(f, self.vertices)
        with open(os.path.join(output_dir, "label.npy"), "wb") as f:
            np.save(f, self.labels)

        print("Dump %d instances in total" % (len(self.labels)))

        self.adj_matrices = []
        self.features = []
        self.vertices = []
        self.labels = []
        
    def dump(self):
        print("Dump data ...")
        graph = self.graph
        diffusion = self.diffusion

        nu = 0
        for cascade_idx, cascade in diffusion.items():
            nu += 1
            if nu % 1000 == 0:
                print("%d (%.2f percent) diffusion processed" % (nu, 100.*nu/len(diffusion)))

            if len(cascade)<min_inf or len(cascade)>=max_inf:
                continue
            user_affected_all = set([item[0] for item in cascade])
            user_affected_now = set()
            last = 0
            #infected = set((cas[0][0],))
            for item in cascade[1:]:
                u, t = item
                while last < len(cascade) and cascade[last][1] < t:
                    user_affected_now.add(cascade[last][0])
                    last += 1
                if len(user_affected_now) == 0:
                    continue
                if u in user_affected_now:
                    continue
                degree_u=self.degree[u]
                if degree_u>=min_degree and degree_u<max_degree:
                    # create positive case for user u, photo p, time t
                    self.create(u, cascade_idx, t, 1, user_affected_now)

                neighbors_u=set(graph.neighbors(u))
                negative = list(neighbors_u - user_affected_all)
                negative = [v for v in negative \
                        if self.degree[v]>=min_degree \
                        and self.degree[v]<max_degree]
                if len(negative) == 0:
                    continue
                negative_sample = np.random.choice(negative,
                        size=min(argsNegative, len(negative)), replace=False)
                for v in negative_sample:
                    # create negative case for user v photo p, time t
                    self.create(v, cascade_idx, t, 0, user_affected_now)
        if len(self.labels) > 0:
            self.dump_data()


digg_data = Digg()
digg_data.load_graph()
digg_data.load_vote()

digg_data.sort_diffusion()
digg_data.summarize_diffusion()
digg_data.summarize_graph()
digg_data.compute_structural_features()
#digg_data.dump()
    
diff = digg_data.diffusion
diff_size = digg_data.diffusion_size
graph_degree=digg_data.degree
node_fea = digg_data.structural_features
print("Done.")

x= digg_data.extract_edge_list(graph_file)





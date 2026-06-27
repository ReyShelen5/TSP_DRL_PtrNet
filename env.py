import torch
import numpy as np
import math
import itertools
import matplotlib.pyplot as plt

def get_2city_distance(n1, n2):
    x1, y1, x2, y2 = n1[0], n1[1], n2[0], n2[1]
    if isinstance(n1, torch.Tensor):
        return torch.sqrt((x2-x1).pow(2)+(y2-y1).pow(2))
    elif isinstance(n1, (list, np.ndarray)):
        return math.sqrt(pow(x2-x1,2)+pow(y2-y1,2))
    else:
        raise TypeError
    
class Env_tsp():
    # --- CHANGED: Added custom_nodes argument ---
    def __init__(self, cfg, custom_nodes=None):
        '''
        custom_nodes: Can be a single set of coordinates (city_t, 2) 
                      or a pre-made batch (batch, city_t, 2)
        '''
        self.batch = cfg.batch
        self.city_t = cfg.city_t
        
        # Move custom nodes to the correct device if provided
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        if custom_nodes is not None:
            if not isinstance(custom_nodes, torch.Tensor):
                self.custom_nodes = torch.tensor(custom_nodes, dtype=torch.float32, device=self.device)
            else:
                self.custom_nodes = custom_nodes.to(self.device)
        else:
            self.custom_nodes = None
            
    # --- CHANGED: Uses custom nodes if available ---
    def get_nodes(self, seed = None):
        '''
        return nodes:(city_t,2)
        '''
        if self.custom_nodes is not None:
            # If a full batch was given, return the first one, otherwise return the whole 2D tensor
            if self.custom_nodes.dim() == 3:
                return self.custom_nodes[0]
            return self.custom_nodes
            
        if seed is not None:
            torch.manual_seed(seed)
        return torch.rand((self.city_t, 2), device = self.device)
        
    def stack_nodes(self):
        '''
        nodes:(city_t,2)
        return inputs:(batch,city_t,2)
        '''
        # --- CHANGED: If a 3D batch was already provided, return it directly ---
        if self.custom_nodes is not None and self.custom_nodes.dim() == 3:
            return self.custom_nodes
            
        list = [self.get_nodes() for i in range(self.batch)]
        inputs = torch.stack(list, dim = 0)
        return inputs
    
    # --- CHANGED: Uses custom nodes if available ---
    def get_batch_nodes(self, n_samples, seed = None):
        '''
        return nodes:(batch,city_t,2)
        '''
        if self.custom_nodes is not None:
            if self.custom_nodes.dim() == 3:
                return self.custom_nodes[:n_samples]
            else:
                # Repeat the single city matrix across the batch dimension
                return self.custom_nodes.unsqueeze(0).repeat(n_samples, 1, 1)

        if seed is not None:
            torch.manual_seed(seed)
        return torch.rand((n_samples, self.city_t, 2), device = self.device)
        
    def stack_random_tours(self):
        list = [self.get_random_tour() for i in range(self.batch)]
        tours = torch.stack(list, dim = 0)
        return tours
        
    def stack_l(self, inputs, tours):
        list = [self.get_tour_distance(inputs[i], tours[i]) for i in range(self.batch)]
        l_batch = torch.stack(list, dim = 0)
        return l_batch

    def stack_l_fast(self, inputs, tours):
        d = torch.gather(input = inputs, dim = 1, index = tours[:,:,None].repeat(1,1,2))
        return (torch.sum((d[:, 1:] - d[:, :-1]).norm(p = 2, dim = 2), dim = 1)
                + (d[:, 0] - d[:, -1]).norm(p = 2, dim = 1))
    
    def show(self, nodes, tour):
        nodes = nodes.cpu().detach()
        tour = tour.cpu().detach()

        distance = self.get_tour_distance(nodes, tour)

        print("\n===================================")
        print("TOUR INFORMATION")
        print("===================================")
        print(f"Total Distance : {distance:.4f}\n")

        print("Input Cities")
        print("-----------------------------------")
        for i in range(self.city_t):
            print(f"City {i:2d} : ({nodes[i,0]:.4f}, {nodes[i,1]:.4f})")

        print("\nPredicted Tour")
        print("-----------------------------------")
        for step, city in enumerate(tour):
            x = nodes[city,0].item()
            y = nodes[city,1].item()

            print(
                f"Step {step+1:2d} -> "
                f"City {city.item():2d} "
                f"({x:.4f}, {y:.4f})"
            )

        print("\nTour Sequence:")
        print(" -> ".join(str(c.item()) for c in tour))

        plt.figure(figsize=(8,8))

        plt.plot(nodes[:,0], nodes[:,1], 'yo', markersize=10)

        plt.plot(nodes[tour,0], nodes[tour,1], 'k-', linewidth=1)

        plt.plot(
            [nodes[tour[-1],0], nodes[tour[0],0]],
            [nodes[tour[-1],1], nodes[tour[0],1]],
            'k-',
            linewidth=1
        )

        for i in range(self.city_t):
            plt.text(
                nodes[i,0],
                nodes[i,1],
                f"{i}\n({nodes[i,0]:.2f},{nodes[i,1]:.2f})",
                fontsize=8,
                color="blue",
                ha="left",
                va="bottom"
            )

        plt.title(f"TSP Tour (Distance = {distance:.4f})")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.show()
    
    def shuffle(self, inputs):
        shuffle_inputs = torch.zeros(inputs.size())
        for i in range(self.batch):
            perm = torch.randperm(self.city_t)
            shuffle_inputs[i,:,:] = inputs[i,perm,:]
        return shuffle_inputs
        
    def back_tours(self, pred_shuffle_tours, shuffle_inputs, test_inputs, device):
        pred_tours = []
        for i in range(self.batch):
            pred_tour = []
            for j in range(self.city_t):
                xy_temp = shuffle_inputs[i, pred_shuffle_tours[i, j]].to(device)
                for k in range(self.city_t):
                    if torch.all(torch.eq(xy_temp, test_inputs[i,k])):
                        pred_tour.append(torch.tensor(k))
                        if len(pred_tour) == self.city_t:
                            pred_tours.append(torch.stack(pred_tour, dim = 0)) 
                        break
        pred_tours = torch.stack(pred_tours, dim = 0)
        return pred_tours 
            
    def get_tour_distance(self, nodes, tour):
        l = 0
        for i in range(self.city_t):
            l += get_2city_distance(nodes[tour[i]], nodes[tour[(i+1)%self.city_t]])
        return l

    def get_random_tour(self):
        tour = []
        while set(tour) != set(range(self.city_t)):
            city = np.random.randint(self.city_t)
            if city not in tour:
                tour.append(city)
        tour = torch.from_numpy(np.array(tour))
        return tour
        
    def get_optimal_tour(self, nodes):
        points = nodes.cpu().numpy() # Added .cpu() to prevent tensor device errors
        all_distances = [[get_2city_distance(x, y) for y in points] for x in points]
        A = {(frozenset([0, idx + 1]), idx + 1): (dist, [0, idx + 1]) for idx, dist in enumerate(all_distances[0][1:])}
        cnt = len(points)
        for m in range(2, cnt):
            B = {}
            for S in [frozenset(C) | {0} for C in itertools.combinations(range(1, cnt), m)]:
                for j in S - {0}:
                    B[(S, j)] = min([(A[(S - {j}, k)][0] + all_distances[k][j], A[(S - {j}, k)][1] + [j]) for k in S if
                                     k != 0 and k != j])
            A = B
        res = min([(A[d][0] + all_distances[0][d[1]], A[d][1]) for d in iter(A)])
        tour = torch.from_numpy(np.array(res[1]))
        return tour


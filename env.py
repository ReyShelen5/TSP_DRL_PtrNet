import torch
import numpy as np
import math
import itertools
import matplotlib.pyplot as plt
import pandas as pd

def get_2city_distance(n1, n2):
    """Calculates the Haversine distance (in kilometers) between two coordinate pairs."""
    # n1 and n2 are expected to be [latitude, longitude]
    lat1, lon1, lat2, lon2 = n1[0], n1[1], n2[0], n2[1]
    R = 6371.0  # Earth's radius in kilometers

    if isinstance(n1, torch.Tensor):
        # Convert degrees to radians
        lat1_rad = torch.deg2rad(lat1)
        lon1_rad = torch.deg2rad(lon1)
        lat2_rad = torch.deg2rad(lat2)
        lon2_rad = torch.deg2rad(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = torch.sin(dlat / 2).pow(2) + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon / 2).pow(2)
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
        return R * c

    elif isinstance(n1, (list, np.ndarray)):
        # Convert degrees to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c
    else:
        raise TypeError
    
class Env_tsp():
    def __init__(self, cfg, custom_nodes=None):
        self.batch = cfg.batch
        self.city_t = cfg.city_t
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.bounds = None 
        
        if isinstance(custom_nodes, str):
            # If loading from CSV, normalize after reading
            raw_coords = self.load_real_coordinates(custom_nodes)
            self.custom_nodes = self.normalize_nodes(raw_coords)
        elif custom_nodes is not None:
            if not isinstance(custom_nodes, torch.Tensor):
                raw_coords = torch.tensor(custom_nodes, dtype=torch.float32, device=self.device)
            else:
                raw_coords = custom_nodes.to(self.device)
            
            # Normalize user-entered coordinates
            self.custom_nodes = self.normalize_nodes(raw_coords)
        else:
            self.custom_nodes = None
    def load_real_coordinates(self, csv_path):
        df = pd.read_csv(csv_path)

        coords = torch.tensor(
            df[['x', 'y']].values,
            dtype=torch.float32,
            device=self.device
        )

        if coords.shape[0] != self.city_t:
            raise ValueError(
                f"CSV contains {coords.shape[0]} cities, expected {self.city_t}"
            )

        return coords

    def get_nodes(self, seed = None):
        if self.custom_nodes is not None:
            if self.custom_nodes.dim() == 3:
                return self.custom_nodes[0]
            return self.custom_nodes
            
        if seed is not None:
            torch.manual_seed(seed)
            
        # 1. Generate realistic coordinates within Santa Catarina's box
        # Latitudes: -28.5 to -26.0 | Longitudes: -54.0 to -48.5
        lats = -28.5 + torch.rand((self.city_t, 1), device=self.device) * 2.5
        lons = -54.0 + torch.rand((self.city_t, 1), device=self.device) * 5.5
        raw_nodes = torch.cat((lats, lons), dim=1)
        
        # 2. Normalize them to [0, 1] for the neural network
        return self.normalize_nodes(raw_nodes)

    def normalize_nodes(self, nodes):
        """Normalizes raw lat/lon nodes to [0, 1] and saves boundaries."""
        min_vals = nodes.min(dim=0, keepdim=True)[0]
        max_vals = nodes.max(dim=0, keepdim=True)[0]
        
        # Save bounds so stack_l_fast can denormalize them for Haversine math
        self.bounds = {'min': min_vals, 'max': max_vals}
        
        denom = max_vals - min_vals
        denom[denom == 0] = 1.0
        return (nodes - min_vals) / denom

    def denormalize_nodes(self, normalized_nodes):
        """Transforms [0, 1] nodes back to actual raw Lat/Lon degrees."""
        if self.bounds is None:
            return normalized_nodes 
        return normalized_nodes * (self.bounds['max'] - self.bounds['min']) + self.bounds['min']
        
    def stack_nodes(self):
        if self.custom_nodes is not None and self.custom_nodes.dim() == 3:
            return self.custom_nodes
            
        list = [self.get_nodes() for i in range(self.batch)]
        inputs = torch.stack(list, dim = 0)
        return inputs
    
    def get_batch_nodes(self, n_samples, seed = None):
        if self.custom_nodes is not None:
            if self.custom_nodes.dim() == 3:
                return self.custom_nodes[:n_samples]
            else:
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
        """Vectorized batch calculation of Haversine distance loops."""
        # Gather the normalized coordinates selected by the network
        d_norm = torch.gather(input = inputs, dim = 1, index = tours[:,:,None].repeat(1,1,2))
        
        # Denormalize them to actual Earth degrees for Haversine math
        d = self.denormalize_nodes(d_norm)
        
        R = 6371.0
        d_next = torch.roll(d, shifts=-1, dims=1)

        lat1 = torch.deg2rad(d[:, :, 0])
        lon1 = torch.deg2rad(d[:, :, 1])
        lat2 = torch.deg2rad(d_next[:, :, 0])
        lon2 = torch.deg2rad(d_next[:, :, 1])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = torch.sin(dlat / 2).pow(2) + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2).pow(2)
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
        return torch.sum(R * c, dim=1)
    
    def show(self, nodes, tour):
        nodes = nodes.cpu().detach()
        tour = tour.cpu().detach()

        real_nodes = self.denormalize_nodes(nodes)
        distance = self.get_tour_distance(real_nodes, tour)

        print("\n===================================")
        print("TOUR INFORMATION")
        print("===================================")
        print(f"Total Distance : {distance:.4f} KM\n")

        print("Input Cities")
        print("-----------------------------------")
        for i in range(self.city_t):
            print(f"City {i:2d} : ({real_nodes[i,0]:.4f}, {real_nodes[i,1]:.4f})")

        print("\nPredicted Tour")
        print("-----------------------------------")
        for step, city in enumerate(tour):
            x = real_nodes[city,0].item()
            y = real_nodes[city,1].item()

            print(
                f"Step {step+1:2d} -> "
                f"City {city.item():2d} "
                f"({x:.4f}, {y:.4f})"
            )

        print("\nTour Sequence:")
        print(" -> ".join(str(c.item()) for c in tour))

        plt.figure(figsize=(8,8))
        plt.plot(real_nodes[:,0], real_nodes[:,1], 'yo', markersize=10)
        plt.plot(real_nodes[tour,0], real_nodes[tour,1], 'k-', linewidth=1)
        plt.plot(
            [real_nodes[tour[-1],0], real_nodes[tour[0],0]],
            [real_nodes[tour[-1],1], real_nodes[tour[0],1]],
            'k-',
            linewidth=1
        )

        for i in range(self.city_t):
            plt.text(
                real_nodes[i,0],
                real_nodes[i,1],
                f"{i}\n({real_nodes[i,0]:.2f},{real_nodes[i,1]:.2f})",
                fontsize=8,
                color="blue",
                ha="left",
                va="bottom"
            )

        plt.title(f"TSP Tour (Distance = {distance:.4f} KM)")
        plt.xlabel("Latitude")
        plt.ylabel("Longitude")
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
        points = nodes.cpu().numpy() 
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
import numpy as np
from .optimized import compute_distances
from itertools import groupby

def signed_area(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the signed area of a polygon defined by vertices (x, y)."""
    return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))

class Loop:
    """This class represents a 2D Polygon loop defined by its x and y coordinates.
    """
    def __init__(self, xs: np.ndarray, ys: np.ndarray):
        self.x: np.ndarray = np.array(xs)
        self.y: np.ndarray = np.array(ys)
        if self.x[0]==self.x[-1]:
            self.x = self.x[:-1]
            self.y = self.y[:-1]
        self.N: int = self.x.shape[0]
        self.solid: bool = False

    def cleanup(self) -> None:
        """
        Remove consecutive duplicate points where both x[i+1] == x[i] and y[i+1] == y[i].
        Updates self.x, self.y, and self.N.
        """
        if self.N == 0:
            return

        mask = np.ones(self.N, dtype=bool)
        mask[1:] = ~((self.x[1:] == self.x[:-1]) & (self.y[1:] == self.y[:-1]))

        self.x = self.x[mask]
        self.y = self.y[mask]
        self.N = self.x.shape[0]
    
    def split(self, merge_margin: float = 1e-9) -> tuple[list[tuple[list[float],list[float]]], list[tuple[list[float],list[float]]]]:
        """Splits the loop into an Add and Remove list of (x,y) points.

        Args:
            merge_margin (float, optional): The distance below which points are considered the same. Defaults to 1e-9.

        Returns:
            tuple[list[tuple[list[float],list[float]]], list[tuple[list[float],list[float]]]]: A tuple containing two lists:
                - The first list contains loops to be added, each represented as a tuple of (x_points, y_points).
                - The second list contains loops to be removed, each represented as a tuple of (x_points, y_points).
        """
        self.cleanup()
        self.unique = np.ones_like(self.x, dtype=np.int32)
        
        ds = compute_distances(self.x, self.y, 0*self.x)
        
        conn = dict()
        # For each polygon node index
        for i in range(self.N):
            # get a list of each connected node that is not itself.
            dsv = np.concatenate((ds[i,:i],[1.0,],ds[i,i+1:]))
            # Truth map of those that are the same
            connected = dsv < merge_margin
            # Map node id -> list of all nodes that are the same [j, k, l]
            conn[i] = list(np.argwhere(connected).flatten())
            # If at least one node is the same, set all unique flags to 0
            if len(conn[i])>0:
                self.unique[np.argwhere(connected)] = 0
        
        # if there are no similar nodes, just return the loop
        if sum(self.unique)==self.unique.shape[0]:
            return ([(self.x, self.y),], [])

        
        id_loop = np.zeros_like(self.x)
        for i, c in conn.items():
            if len(c)!=0:
                id_loop[i]=-1
        
        loopid = 1
        loops = []
        while True:
            if np.all(id_loop != 0):
                break

            start_id = int(np.where(id_loop==0)[0][0])
            id_loop[start_id] = loopid
            current_loop = [start_id,]
            pause_till = -1
            for i in range(start_id+1, self.N):
                if id_loop[i]>0:
                    continue
                if i < pause_till:
                    continue
                if i == pause_till:
                    pause_till = -1
                    continue
                
                if len(conn[i])==0:
                    current_loop.append(i)
                    id_loop[i] = loopid
                else:
                    connected_to = conn[i]
                    if connected_to[0] == current_loop[0]-1:
                        current_loop.append(i)
                        id_loop[i] = loopid
                        break
                    largest = connected_to[-1]
                    current_loop.append(i)
                    id_loop[i] = loopid
                    pause_till = largest
            loopid += 1
            loops.append(current_loop)
        
        areas = [signed_area(self.x[loop], self.y[loop]) for loop in loops]
        np_areas = np.array(areas)
        
        add_loops = []
        remove_loops = []
        
        sign_biggest = np.sign(np_areas[0])
        
        for (loop, A) in zip(loops, areas):
            
            if np.sign(A)*sign_biggest > 0:
                add_loops.append(loop)
            else:
                remove_loops.append(loop)
        
        output_add = [(self.x[loop], self.y[loop]) for loop in add_loops]
        output_remove = [(self.x[loop], self.y[loop]) for loop in remove_loops]
        
        return output_add, output_remove
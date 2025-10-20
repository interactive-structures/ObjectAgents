import numpy as np
import heapq
from collections.abc import Callable

# primitives are in the form:
# delta x, delta y, new orientation, cost
def generate_simple_primitives(step_size=10, n_movement_options:int=8):
    primitives = []
    for i in range(n_movement_options):
        theta = i/n_movement_options * 2 * np.pi
        dx = step_size * np.cos(theta)
        dy = step_size * np.sin(theta)
        primitives.append( (dx, dy, 0, step_size) )
    return [primitives]

def euclidean_distance(ax, ay, bx, by):
    return np.hypot(ax - bx, ay - by)

def sq_euclidean_distance(ax, ay, bx, by):
    """Avoid expensive sqrt operation by using squared distance"""
    return (ax - bx)**2 + (ay - by)**2

def neg_sq_euclidean_distance(ax, ay, bx, by):
    return -1 * sq_euclidean_distance(ax, ay, bx, by)


def make_goal_NEAR(target_x, target_y, threshold):
    threshold_squared = threshold**2
    def goal_function(x, y):
        current_distance = sq_euclidean_distance(x, y, target_x, target_y)
        return current_distance <= threshold_squared
    return goal_function

def make_goal_AWAY(target_x, target_y, threshold):
    threshold_squared = threshold**2
    def goal_function(x, y):
        current_distance = sq_euclidean_distance(x, y, target_x, target_y)
        return current_distance >= threshold_squared
    return goal_function

def make_heuristic_NEAR(target_x, target_y):
    def heuristic(x, y):
        return sq_euclidean_distance(x, y, target_x, target_y)
    return heuristic
    
def make_heuristic_AWAY(target_x, target_y):
    def heuristic(x, y):
        return neg_sq_euclidean_distance(x, y, target_x, target_y)
    return heuristic


def astar_with_motion_primitives(
        start_x: float,
        start_y: float,
        start_theta: float,
        goal: Callable[[float, float], bool],
        heuristic: Callable[[float, float], float],
        map_data: np.ndarray,
        primitives: list,
        n_orientations: int = 1,
        timeout = 2000
        ):
    
    start_theta_bin = int((start_theta / (2 * np.pi) * n_orientations)) % n_orientations
    
    # Initialize search
    open_set = []
    closed_set = set()
    g_score = {}
    came_from = {}
    
    # Create start node
    start_node = (start_x, start_y, start_theta_bin)
    g_score[start_node] = 0
    f_score = g_score[start_node] + heuristic(start_x, start_y)

    insertion_counter = 0 # used as tiebreaker when retrieving from heap
    heapq.heappush(open_set, (f_score, insertion_counter, start_node))  
    
    iteration = 0
    
    # Main search loop
    while open_set:
        _, _, current = heapq.heappop(open_set)
        x, y, theta_bin = current
        
        # Check if reached goal region
        if goal(x, y):
            raw_path = reconstruct_path(current, came_from)
            return raw_path

        # Exit if we've been searching for too long
        iteration += 1
        if iteration > timeout:
            if heuristic(current[0], current[1]) < heuristic(start_x, start_y):
                raw_path = reconstruct_path(current, came_from)
                return raw_path
            else:
                return None
        
        # Get primitives for current orientation bin
        available_primitives = primitives[theta_bin]

        for (dx, dy, new_theta_bin, cost) in available_primitives:
            new_x, new_y = int(round(x + dx)), int(round(y + dy))

            # If we went out of bounds, continue searching
            if not (0 <= new_y < map_data.shape[0] and 0 <= new_x < map_data.shape[1]):
                continue 

            # If an obstacle is present, then continue searching through primitives
            if map_data[new_y, new_x] > 0:
                continue
        
            # Create new node
            neighbor = (new_x, new_y, new_theta_bin)
            
            # Skip if already processed
            if neighbor in closed_set:
                continue

            # Calculate costs
            tentative_g = g_score.get(current) + cost

            # could also add additional cost for being close to obstacles    
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                # Update path
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g

                # TODO: keep track of the action (motion primitive) that we chose here
                
                # Calculate f_score
                f_score = tentative_g + heuristic(new_x, new_y)
                insertion_counter += 1
                heapq.heappush(open_set, (f_score, insertion_counter, neighbor))

        # Add to closed set (node has been explored)
        closed_set.add(current)
    
    # No path found, return closest
    if heuristic(current[0], current[1]) < heuristic(start_x, start_y):
        raw_path = reconstruct_path(current, came_from)
        return raw_path
    else:
        print("routefinding: No path found")
        return None

# def astar_with_motion_primitives(
#         start_x: float,
#         start_y: float,
#         start_theta: float,
#         goal_x: float,
#         goal_y: float,
#         map_data: np.ndarray,
#         primitives: list,
#         n_orientations: int = 1,
#         heuristic: Callable[[float, float, float, float], float] = sq_euclidean_distance,
#         timeout = 2000
#         ):
    
#     start_theta_bin = int((start_theta / (2 * np.pi) * n_orientations)) % n_orientations
    
#     # Initialize search
#     open_set = []
#     closed_set = set()
#     g_score = {}
#     came_from = {}
    
#     # Create start node
#     start_node = (start_x, start_y, start_theta_bin)
#     g_score[start_node] = 0
#     f_score = g_score[start_node] + heuristic(start_x, start_y, goal_x, goal_y)

#     insertion_counter = 0 # used as tiebreaker when retrieving from heap
#     heapq.heappush(open_set, (f_score, insertion_counter, start_node))  
    
#     # Define goal region
#     goal_tolerance = 50
#     sq_goal_tolerance = goal_tolerance**2

#     iteration = 0
    
#     # Main search loop
#     while open_set:
#         _, _, current = heapq.heappop(open_set)
#         x, y, theta_bin = current
        
#         # Check if reached goal region
#         current_distance = sq_euclidean_distance(x, y, goal_x, goal_y)
#         if current_distance <= sq_goal_tolerance:
#             raw_path = reconstruct_path(current, came_from)
#             return raw_path

#         # If we've been searching for too long, return a partial path (if we've found one)
#         iteration += 1
#         if iteration > timeout:
#             if current_distance < sq_euclidean_distance(start_x, start_y, goal_x, goal_y):
#                 raw_path = reconstruct_path(current, came_from)
#                 return raw_path
#             else:
#                 return None
        
#         # Get primitives for current orientation bin
#         available_primitives = primitives[theta_bin]

#         for (dx, dy, new_theta_bin, cost) in available_primitives:
#             new_x, new_y = int(round(x + dx)), int(round(y + dy))

#             # If we went out of bounds, continue searching
#             if not (0 <= new_y < map_data.shape[0] and 0 <= new_x < map_data.shape[1]):
#                 continue 

#             # If an obstacle is present, then continue searching through primitives
#             if map_data[new_y, new_x] > 0:
#                 continue
        
#             # Create new node
#             neighbor = (new_x, new_y, new_theta_bin)
            
#             # Skip if already processed
#             if neighbor in closed_set:
#                 continue

#             # Calculate costs
#             tentative_g = g_score.get(current) + cost

#             # could also add additional cost for being close to obstacles    
            
#             if neighbor not in g_score or tentative_g < g_score[neighbor]:
#                 # Update path
#                 came_from[neighbor] = current
#                 g_score[neighbor] = tentative_g

#                 # TODO: keep track of the action (motion primitive) that we chose here
                
#                 # Calculate f_score
#                 f_score = tentative_g + heuristic(new_x, new_y, goal_x, goal_y)
#                 insertion_counter += 1
#                 heapq.heappush(open_set, (f_score, insertion_counter, neighbor))

#         # Add to closed set (node has been explored)
#         closed_set.add(current)
    
#     # No path found
#     return None

# deprecated 
# moved to _draw_paths_on_frame in display.py
def plot_path(path, map_data):
    """Visualize the map with the path overlaid."""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.imshow(map_data, cmap='gray', vmin=0, vmax=1)
    
    # Extract x and y coordinates from path
    x_coords = [p[0] for p in path]
    y_coords = [p[1] for p in path]
    
    # Plot path as a red line
    plt.plot(x_coords, y_coords, 'r-', linewidth=2)
    plt.title('Map with Path (White=Free, Black=Obstacle)')
    plt.show()

def reconstruct_path(current, came_from):
    path = []
    
    # Add the current (goal) node
    x, y, _ = current
    path.append((x, y))
    
    # Follow parent links back to start
    while current in came_from:
        current = came_from[current]
        x, y, _ = current
        path.append((x, y))
    
    # Reverse to get path from start to goal
    return path[::-1]


def line_of_sight_optimization(raw_path, map_data):
    """
    Optimize a path using line-of-sight optimization.
    
    Args:
        raw_path: List of waypoints from A* algorithm
        map_data: 2D numpy array where 0 represents obstacles and 1 represents free space
        
    Returns:
        List of (x, y) tuples representing the optimized path
    """
    if not raw_path or len(raw_path) < 3:
        return raw_path  # Nothing to optimize if path is empty or has fewer than 3 points
    
    # Extract only the x, y coordinates
    path_coords = [(p[0], p[1]) for p in raw_path]
        
    optimized_path = [path_coords[0]]  # Start with the first point
    current_idx = 0
    
    while current_idx < len(path_coords) - 1:
        # Find the furthest visible point from current point
        furthest_visible_idx = current_idx + 1
        
        for i in range(current_idx + 2, len(path_coords)):
            if has_line_of_sight(path_coords[current_idx], path_coords[i], map_data):
                furthest_visible_idx = i
        
        # Add the furthest visible point to optimized path
        optimized_path.append(path_coords[furthest_visible_idx])
        
        # Move to the furthest visible point
        current_idx = furthest_visible_idx
    
    return optimized_path

def has_line_of_sight(point1, point2, map_data):
    """
    Check if there is a clear line of sight between two points.
    
    Args:
        point1: Starting point (x, y)
        point2: Ending point (x, y)
        map_data: 2D numpy array where 0 represents obstacles and 1 represents free space
        
    Returns:
        Boolean indicating if there is a clear line of sight
    """
    x1, y1 = point1
    x2, y2 = point2
    
    # Use Bresenham's algorithm to find grid cells along the line
    cells = bresenham_line(x1, y1, x2, y2)
    
    # Check if any cell along the line is an obstacle
    for x, y in cells:
        # Check if the point is within map bounds
        if 0 <= y < map_data.shape[0] and 0 <= x < map_data.shape[1]:
            if map_data[y, x] > 0:  # obstacle
                return False
        else:
            return False  # Point outside map bounds
            
    return True

def bresenham_line(x1, y1, x2, y2):
    """
    Bresenham's line algorithm to find grid cells that a line passes through.
    
    Args:
        x1, y1: Starting point
        x2, y2: Ending point
        
    Returns:
        List of (x, y) tuples representing grid cells the line passes through
    """
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    while True:
        points.append((x1, y1))
        
        if x1 == x2 and y1 == y2:
            break
            
        e2 = 2 * err
        if e2 > -dy:
            if x1 == x2:
                break
            err -= dy
            x1 += sx
        if e2 < dx:
            if y1 == y2:
                break
            err += dx
            y1 += sy
            
    return points
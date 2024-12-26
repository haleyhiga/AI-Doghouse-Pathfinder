import heapq



class DogModel:
    def __init__(self):

        self.agent_location = None
        self.target_location = None
        self.grid_size = 15  # grid size
        self.obstacles = set()  # set of obstacles

    def reset(self):
        """Reset the model state."""
        self.agent_location = (0, 0)
        self.target_location = (4, 4)
        self.obstacles = set()

    def update(self, state):
        """Update the agent's state."""
        self.agent_location = tuple(state["agent"])
        self.target_location = tuple(state["target"])
        self.obstacles = set(map(tuple, state["obstacles"]))

    def ACTIONS(self, s):
        """Return the list of valid actions in state `s`."""
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # RIGHT, UP, LEFT, DOWN
        
        neighbors = []  
        for d in directions:  
            neighbor = (s[0] + d[0], s[1] + d[1])
            neighbors.append(neighbor)
        
        valid_neighbors = [] 
        for n in neighbors:
            if (0 <= n[0] < self.grid_size and 0 <= n[1] < self.grid_size and n not in self.obstacles):
                valid_neighbors.append(n)
        
        return valid_neighbors


    def RESULT(self, s, a):
        return a  # In this case, the action `a` is the next state.

    def GOAL_TEST(self, s):
        return s == self.target_location

    def STEP_COST(self, s, a, s1):
        return 1  # same cost for all movements.

    def HEURISTIC(self, s):
        x_distance = abs(s[0] - self.target_location[0])  # delta x
        y_distance = abs(s[1] - self.target_location[1])  # delta y
        total_distance = x_distance + y_distance         
        return total_distance


    def a_star_search(self, start, goal):
        open_list = []
        closed_list = set()
        heapq.heappush(open_list, (0 + self.HEURISTIC(start), 0, start, []))

        while open_list:
            _, g, current, path_so_far = heapq.heappop(open_list)

            if current == goal:
                return path_so_far + [current]

            if current in closed_list:
                continue

            closed_list.add(current)

            for neighbor in self.ACTIONS(current):
                if neighbor not in closed_list:
                    f = g + self.STEP_COST(current, None, neighbor) + self.HEURISTIC(neighbor)
                    heapq.heappush(
                        open_list, (f, g + 1, neighbor, path_so_far + [current])
                    )

        return []  # didnt find a path

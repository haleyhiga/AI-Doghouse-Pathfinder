Doggy Environment
-------------------------


# Description

In this environment, a state is made up positions of the agent, 
target, and all the obstacles at the moment. 
 All of these positions are represented as x and y coordinates.
   The goal is for the dog to get to the doghouse efficiently and avoid all obstacles.

# Observation Space

An observation is a dictionary where each value in that dictionary is a numpy.array of integers.
The keys in the dictionary are "agent", "target", and "obstacles."  
The shape of the agent numpy.array is (2,) for x and y coordinates.  
The shape of the target is also of shape (2,) for x and y coordinates.  
And the shape of obstacles is of (num_obstacles, 2), for the number of obstacles in the environment and x, y coordinates.

# Action Space

An action is an integer from 0 to 3 that represents right, up, left, or down.  There are four discrete actions: 0 being right, 1 is up, 2 is left, and 3 is down.

# Starting State

The starting state randomly places the agent at a position inside of the grid.  The target and obstacle locations are also placed at a random position within the grid, making sure that they are not allowed to be placed at the same area as each other.

# Rewards

The agent will receive a reward of 1 if it reaches the target location.  However, if it does not reach the location within a certain amount of time, it will receive a reward of 0.


# Episode End

The episode terminates if the agent reaches the target.  The maximum episode steps can be configured.
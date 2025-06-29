# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from collections import deque
import heapq

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    from game import Directions

    # Start state
    start_state = problem.getStartState()
    
    # Stack for DFS (current_position, hit_wall_count, path_taken)
    stack = [(start_state, 0, [])]

    # Visited set (position, wall_hits)
    visited = set()
    visited.add((start_state, 0))

    while stack:
        # Pop the last added node (LIFO)
        current_state, hit_walls, current_actions = stack.pop()

        # If the number of wall hits is greater than 2, skip this path
        if hit_walls > 2:
            continue

        # Checks if destination is reached
        if problem.isGoalState(current_state):
            return current_actions

        # Expand the node
        for successor, action, _ in problem.getSuccessors(current_state):
            if problem.isWall(successor):
                next_state = (successor, hit_walls + 1)  # Increase wall hit count
            else:
                next_state = (successor, hit_walls)  # Maintain current wall hit count

            # If the next state (position, wall_hits) has not been visited, add to stack
            if next_state not in visited:
                stack.append((next_state[0], next_state[1], current_actions + [action]))
                visited.add(next_state)

    return []  # If no solution found

def breadthFirstSearch(problem):
    from game import Directions

    # Initialize queue for BFS (current_position, wall_hit_count, path_taken)
    queue = deque([(problem.getStartState(), 0, [])])

    # Visited set (position, wall_hits)
    visited = set()
    visited.add((problem.getStartState(), 0))

    while queue:
        # Get the first element from the queue (FIFO behavior)
        current_state, hit_walls, current_actions = queue.popleft()

        # Stop searching if the wall hit count exceeds 2
        if hit_walls > 2:
            continue

        # Check if Pacman has reached the goal
        if problem.isGoalState(current_state):
            return current_actions

        # Expand the current state
        for successor, action, _ in problem.getSuccessors(current_state):
            # Check if successor is a wall
            if problem.isWall(successor):
                next_state = (successor, hit_walls + 1)  # Increase wall hit count
            else:
                next_state = (successor, hit_walls)  # Maintain current wall hit count

            # If the next state is not visited, add it to the queue
            if next_state not in visited:
                queue.append((next_state[0], next_state[1], current_actions + [action]))
                visited.add(next_state)

    return []  # If no solution found

def uniformCostSearch(problem):
    from game import Directions

    # Start state
    startState = problem.getStartState()

    # Priority queue (cost, current_position, wall_hit_count, path_taken)
    priorityQueue = [(0, startState, 0, [])]  # Initial cost = 0

    # Distance dictionary to track the minimum cost to each node
    distance = {(startState, 0): 0}

    # Visited set 
    visited = set()
    visited.add((startState, 0))

    while priorityQueue:
        # Pop the node with the lowest cost
        cost, currentState, hitWalls, currentActions = heapq.heappop(priorityQueue)

        # If the wall hit count exceeds 2, ignore this path
        if hitWalls > 2:
            continue

        # If we reach the goal, return the path
        if problem.isGoalState(currentState):
            if hitWalls >= 1:  # Must hit at least one wall
                return currentActions
            else:
                continue  # Keep searching if no wall hit

        # Expand the current state
        for successor, action, stepCost in problem.getSuccessors(currentState):
            # Determine new wall hit count
            if problem.isWall(successor):
                nextState = (successor, hitWalls + 1)  # Increase wall hit count
            else:
                nextState = (successor, hitWalls)  # Maintain current wall hit count

            # Compute cost to go
            costToGo = cost + stepCost  # Get cost of currentActions + new stepCost

            # If the current path is better (lower cost) than any previously found path
            if nextState not in distance or costToGo < distance[nextState]:
                distance[nextState] = costToGo  # Update cost
                heapq.heappush(priorityQueue, (costToGo, nextState[0], nextState[1], currentActions + [action]))
                visited.add(nextState)

    return []  # If no solution is found

def nullHeuristic(state, problem=None):
    goal = problem.goal  
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])

def aStarSearch(problem, heuristic=nullHeuristic):
    from game import Directions

    # Start state
    startState = problem.getStartState()

    # Priority queue (cost + heuristic, cost, current_position, wall_hit_count, path_taken)
    priorityQueue = [(heuristic(startState, problem), 0, startState, 0, [])]  # Initial cost = 0

    # Distance dictionary to track the minimum cost to each node, considering wall hits
    distance = {(startState, 0): 0}

    # Visited set 
    visited = set()
    visited.add((startState, 0))

    while priorityQueue:
        # Pop the node with the lowest cost + heuristic value
        _, cost, currentState, hitWalls, currentActions = heapq.heappop(priorityQueue)

        # If we reach the goal but haven't hit a wall at least once, ignore this path
        if problem.isGoalState(currentState):
            if hitWalls >= 1:  # ✅ Must hit at least 1 wall
                return currentActions
            else:
                continue  # Keep searching if no wall hit

        # If the wall hit count exceeds 2, ignore path
        if hitWalls > 2:
            continue

        # Expand the current state
        for successor, action, stepCost in problem.getSuccessors(currentState):
            # Determine new wall hit count
            if problem.isWall(successor):
                new_hit_walls = hitWalls + 1  # Increase wall hit count
            else:
                new_hit_walls = hitWalls  # Maintain current wall hit count

            # Compute the cost to go
            new_cost = cost + stepCost  
            priority = new_cost + heuristic(successor, problem)  

            # If this path is better (lower cost) than any previously found path
            if (successor, new_hit_walls) not in distance or new_cost < distance[(successor, new_hit_walls)]:
                distance[(successor, new_hit_walls)] = new_cost  # Update cost
                heapq.heappush(priorityQueue, (priority, new_cost, successor, new_hit_walls, currentActions + [action]))
                visited.add((successor, new_hit_walls))

    return []  # If no solution is found


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

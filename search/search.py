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
from game import Directions
from typing import List

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """
    # what is truely used are the subclasses of SearchProblem, the abstract class is just an interface

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
        # this action is kinda like expanding the fringe.

        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    fringe = util.Stack() # LIFO policy list
    # fringe starts with start state and the empty action list. 
    fringe.push((problem.getStartState(), [])) 
    visited = set() # since we wanna remove duplicates for the graph search. 

    while not fringe.isEmpty():
        state, actions = fringe.pop() # expand the node 

        if problem.isGoalState(state):
            return actions
        if not (state in visited):
             visited.add(state) # the set will automatically take care of duplicates. 
             for successor, action, _ in problem.getSuccessors(state):
                # decide the next state to visit. 
                # the most recent push wil be the next one to pop. 
                fringe.push((successor, actions + [action])) 

    return []

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    fringe = util.Queue() # FIFO policy list
    fringe.push((problem.getStartState(), [])) # start state and the empty action list. 
    visited = set() # since we wanna remove duplicates for the graph search. 

    while not fringe.isEmpty():
        state, actions = fringe.pop() # expand the node 

        if problem.isGoalState(state):
            return actions
        if not (state in visited):
             visited.add(state) # the set will automatically take care of duplicates. 
             for successor, action, _ in problem.getSuccessors(state):
                # decide the next state to visit. 
                # the most recent push wil be the next one to pop. 
                fringe.push((successor, actions + [action])) 

    return []

def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    fringe = util.PriorityQueue() # The priority will be determined by cost. 
    fringe.push((problem.getStartState(), []), problem.getCostOfActions([]))
    visited = set()

    while not fringe.isEmpty():
        state, actions = fringe.pop()

        if problem.isGoalState(state):
            return actions
        if not (state in visited):
             visited.add(state) # the set will automatically take care of duplicates. 
             for successor, action, _ in problem.getSuccessors(state):
                # decide the next state to visit. 
                # the most recent push wil be the next one to pop. 
                fringe.push((successor, actions + [action]),problem.getCostOfActions(actions + [action])) 
    return []
    

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    fringe = util.PriorityQueue()
    start_state = problem.getStartState()
    
    fringe.push(start_state, heuristic(start_state, problem))
    best_cost = {start_state: 0}
    best_actions = {start_state: []}

    while not fringe.isEmpty():
        state = fringe.pop()

        if problem.isGoalState(state):
            return best_actions[state]
        
        for successor, action, step_cost in problem.getSuccessors(state):
            new_cost = best_cost[state] + step_cost
            if successor not in best_cost or new_cost < best_cost[successor]:
                best_cost[successor] = new_cost
                best_actions[successor] = best_actions[state] + [action]
                priority = new_cost + heuristic(successor, problem)
                fringe.update(successor, priority)

    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

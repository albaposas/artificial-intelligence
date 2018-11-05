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

def generalGraphSearch(problem, data_structure):

    """
    Generic method that performs graph search.

    If data_structure is a stack then DFS is performed. If it's a Queue then
    BFS. Priority can also be defined simply.

    Our data structure consists of tuples of firstly a state, secondly a list
    of states to which we append from its end (this is important!!) that
    represent the path that needs to be followed from the starting position in
    order to get to the end, and lastly a counter for the cost of visiting
    every node.
    """

    nodes = data_structure # this name makes a lot of sense

    expanded_states = list() # refers to the grid
    state = problem.getStartState()
    nodes.push( (state, [], 0) ) # the root is pushed first

   
    while not nodes.isEmpty():
        (state,path,cost) = nodes.pop()

        if problem.isGoalState(state):
            return path
            
        if state not in expanded_states:
            expanded_states.append(state)
            for newState, newMove, newCost in problem.getSuccessors(state):
                nodes.push((newState, path + [newMove], cost + newCost))

    return None
            


def depthFirstSearch(problem):

    return generalGraphSearch( problem, util.Stack() )

def breadthFirstSearch(problem):

    return generalGraphSearch( problem, util.Queue() )

def uniformCostSearch(problem):
    """
    The anonymus function we pass as an argument to the method refers to the
    third element of the tuples (nodes) that are actually the priority (cost)
    needed to be paid in order to reach them.
    """
    return generalGraphSearch( problem, util.PriorityQueueWithFunction( lambda f: f[-1] ) )

def nullHeuristic(state, problem=None):
    """
    A heuristic function underestimates the cost from the current state to the
    nearest goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):

    return generalGraphSearch( problem, util.PriorityQueueWithFunction( lambda f: f[-1] + heuristic(f[0], problem) ) )


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

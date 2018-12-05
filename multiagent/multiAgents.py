# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        score = successorGameState.getScore()
        newGhostPos = successorGameState.getGhostPositions()
        distToGhost = manhattanDistance(newPos, newGhostPos[0])

        """
        It is not a good thing to have a ghost nearby that can kill PAC-MAN.
        """

        if distToGhost <= 3.0 and distToGhost > 0:
            if newScaredTimes[0] == 0:
                score -= 1.0 / distToGhost
            else:
                score += 1.0 / distToGhost

        """
        It is not a good thing to move away from food. Never. Gotta feed yourself.
        """

        if oldFood.asList():
            closestFoodDotDist = min( manhattanDistance(newPos,food)
                                      for food in oldFood.asList() )
            if closestFoodDotDist > 0:
                score += 1.0 / closestFoodDotDist

        return score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    def dispatch(self, gameState, action, turn, alpha, beta):
        """
        We will call turns to the levels of the resulting tree. There are in
        total numberOfAgents * depth turns. At the last turn, the leafs are
        evaluated; at Pacman's, max_value(...) is called; at the ghosts',
        min_value(...). We can deduce the agent index by simple modular
        arithmetic.

        This is a very general method that controls the flow of execution of
        any MultiAgentSearchAgent. Hence, we'll use it for MinimaxAgent
        (disregarding alpha and beta), ExpectimaxAgent (more advanced
        algorithms could be developed by introducing pruning) and AlphaBetaAgent.

        Notice that PACMAN_value refers to max-value in all three possible
        MultiAgentSearchAgent and ghost_value refers to min-value in MinimaxAgent
        and AlphaBetaAgent and to expect-value in ExpectimaxAgent. We're
        using this naming convention in order to keep generality.
        """

        if gameState.isLose() or gameState.isWin():
    		return self.evaluationFunction(gameState)

        turn = (turn + 1) % (gameState.getNumAgents() * self.depth)
        
        if turn == 0:
            return self.evaluationFunction(gameState)

        agentIndex = turn % gameState.getNumAgents()

        if agentIndex == 0:
            return self.PACMAN_value(gameState, turn, alpha, beta)
        else:
            return self.ghost_value(gameState, turn, alpha, beta)

    def generalGetAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction

          This is a very general action selection method for any
          MultiAgentSearchAgent. Hence, we'll use it for both MinimaxAgent
          (disregarding alpha and beta) and AlphaBetaAgent.
        """

        act_val = util.Counter()

        alpha, beta = float("-inf"), float("inf")

        for action in gameState.getLegalActions(self.index):
            successor = gameState.generateSuccessor(self.index, action)
            act_val[action] = self.dispatch( successor, action, self.index,
                                             alpha, beta )
            alpha = max(alpha, act_val[action])

        return act_val.argMax()

    def value_selector(self, gameState, turn, fun):
        """
        Helper method. Given a state and a turn, either evaluate it if the game
        is in win/lose state or evaluate the children nodes according to a
        function (which can be either max(...) or min(...)). More complicated
        functions could improve performance.
        """

        agentIndex = turn % gameState.getNumAgents()

        return fun( self.dispatch( gameState.generateSuccessor(agentIndex, action),
                                   action, turn, 0, 0 )
                    for action in gameState.getLegalActions(agentIndex) )

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)

      ---

      Notice how this algorithm is a special case of Alpha-Beta pruning where
      alpha = beta = 0. This simple fact allows us to reuse the algorithm.
    """

    def PACMAN_value(self, gameState, turn, alpha, beta):
        return self.value_selector(gameState, turn, lambda x: max(x))

    def ghost_value(self, gameState, turn, alpha, beta):
        return self.value_selector(gameState, turn, lambda x: min(x))

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """

        return self.generalGetAction(gameState)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)

      --

      This is the most general algorithm we have developed for scenarios in
      which the agent's move are ALL deterministic.
    """

    def PACMAN_value(self, gameState, turn, alpha, beta):
        agentIndex = turn % gameState.getNumAgents()
        v = float("-inf")

        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            v = max(v, self.dispatch( successor, action, turn, alpha, beta ))
            if v > beta:
                return v
            alpha = max(alpha, v)
        return v

    def ghost_value(self, gameState, turn, alpha, beta):
        agentIndex = turn % gameState.getNumAgents()
        v = float("inf")

        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            v = min(v, self.dispatch( successor, action, turn, alpha, beta ))
            if v < alpha:
                return v
            beta = min(beta, v)
        return v

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        return self.generalGetAction(gameState)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def PACMAN_value(self, gameState, turn, alpha, beta):
        return self.value_selector(gameState, turn, lambda x: max(x))

    def ghost_value(self, gameState, turn, alpha, beta):
        '''
        Now the ghosts are NOT min agents anymore. They now consider the average
        of all successors' values instead of the minimum. This is because the
        model we have of the ghosts' behavior is now probabilistic, i.e. for us
        they can make any action, be it good or bad, effectively yielding a 
        random-looking behavior, so PAC-MAN now considers the expected value
        of their successors. This is the only change.
        '''
        
        agentIndex = turn % gameState.getNumAgents()

        successors = [ gameState.generateSuccessor(agentIndex, action)
                       for action in gameState.getLegalActions(agentIndex) ]

        successors_values = sum( self.dispatch( successor, action, turn, 0, 0 )
                                 for successor in successors )

        return (1.0 / float(len(successors))) * successors_values

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """

        return self.generalGetAction(gameState)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
      
      --
      
      As usual, the evaluation function is just a linear combination of
      features. We'll first extract them and then combine them in a
      meaningful way.
    """

    pacPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    ghostPositions = currentGameState.getGhostPositions()
    newScaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    score = currentGameState.getScore()

    """
    It is not a good thing to have a ghost nearby that can kill PAC-MAN.
    
    We value positively states in which the ghosts are nearby and scared
    since it's in PAC-MAN's interest to eat them. In contrast, we
    value negatively states in which the ghosts are closed and non-scared
    since they can kill PAC-MAN. The worst possible state is then the one
    in which the distance to the ghosts is 0.
    """

    for agentIndex in range(1, currentGameState.getNumAgents()):

        ghost = currentGameState.getGhostState(agentIndex)
        ghostPos = currentGameState.getGhostPosition(agentIndex)
        distToGhost = manhattanDistance(pacPos, ghostPos)

        if distToGhost == 0:
            return float("-inf") # worst possible scenario
        if ghost.scaredTimer == 0:
            score -= 1.0 / distToGhost
        else:
            score += 1.0 / distToGhost

    """
    It is not a good thing to move away from food. Never. Gotta feed yourself.
    
    Here we use the inverse of the distance to the closest food dot there is.
    The closest it is, the higher we value the state, and the farther it is,
    the lower we value the state. This way PAC-MAN can consider as better
    states those that have food the nearest.
    """

    if food:
        closestFoodDotDist = min( manhattanDistance(pacPos, foodDot)
                                  for foodDot in food )
        if closestFoodDotDist > 0:
            score += 1.0 / closestFoodDotDist

    return score

# Abbreviation
better = betterEvaluationFunction

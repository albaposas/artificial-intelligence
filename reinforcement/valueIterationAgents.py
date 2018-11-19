# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        current_values = self.values.copy()

        for k in range(0, iterations):
            for state in self.mdp.getStates():
                if not self.mdp.isTerminal(state):
                    current_values[state] = self.getQValue(state,
                                                           self.getAction(state))
            self.values = current_values.copy()

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """ Q*
          Compute the Q-value of action in state from the
          value function stored in self.values.

          --

          Recall that mdp.getTransitionStatesAndProbs(state, action) returns the
          pair (nextState, prob), where prob is the probability of reaching
          nextState from 'state'.
        """
        q_val = 0

        for T in self.mdp.getTransitionStatesAndProbs(state, action):
            q_val += T[1] * (   self.mdp.getReward(state, action, T[0])
                              + self.discount
                              * self.getValue(T[0]) )

        return q_val

    def computeActionFromValues(self, state):
        """ V*
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.

          ---

          Here we create a Counter that contains all possible actions from
          'state' as keys and all of their Q-values as values. It is then
          trivial to compute the policy, which is just the key with the
          highest value.
        """
        if self.mdp.isTerminal(state):
            return None

        temp_dict = util.Counter()

        for action in self.mdp.getPossibleActions(state):
            temp_dict[action] = self.getQValue(state, action)

        return temp_dict.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

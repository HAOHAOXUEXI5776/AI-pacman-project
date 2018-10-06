# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint


class ValueIterationAgent():
    """
        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.7, iterations = 100):
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

        # Write value iteration code here
        for i in range(0, iterations):
            iteration_values = util.Counter()
            for state in mdp.getStates():
                best_action = self.computeActionFromValues(state)
                if best_action:
                    iteration_values[state] = self.computeQValueFromValues(state, best_action)

            self.values = iteration_values.copy()


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        #print(state, action)
        qvalue = 0
        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, next_state)
            #print prob, reward, self.discount
            qvalue += prob * (reward + self.discount * self.values[next_state])
        return qvalue


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        action_values = util.Counter()
        for action in self.mdp.getPossibleActions(state):
            action_values[action] = self.computeQValueFromValues(state, action)
        return action_values.argMax()


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PacmanMDP():
    def __init__(self, startState, states):
        self._startState = startState
        self._states = states

        legalMoves = util.Counter()
        for state in states:
            x, y = state
            if (x-1, y) in states:
                legalMoves[(state, Directions.WEST)] = (x-1, y)
            if (x+1, y) in states:
                legalMoves[(state, Directions.EAST)] = (x+1, y)
            if (x, y-1) in states:
                legalMoves[(state, Directions.SOUTH)] = (x, y-1)
            if (x, y+1) in states:
                legalMoves[(state, Directions.NORTH)] = (x, y+1)
        self._possibleActions = legalMoves
        self._rewards = util.Counter()

    def addReward(self, state, reward):
        self._rewards[state] += reward

    def addRewardWithNeighbours(self, state, reward):
        x, y = state
        self._rewards[state] += reward
        self._rewards[(x - 1, y)] += reward / 2
        self._rewards[(x + 1, y)] += reward / 2
        self._rewards[(x, y - 1)] += reward / 2
        self._rewards[(x, y + 1)] += reward / 2


    def print_state(self):
        print "States", self._states
        print "Rewards", self._rewards
        print "Possible Actions", self._possibleActions


    def getStates(self):
        """
        Return a list of all states in the MDP.
        Not generally possible for large MDPs.
        """
        return list(self._states)

    def getStartState(self):
        """
        Return the start state of the MDP.
        """
        return self._startState

    def getPossibleActions(self, state):
        """
        Return list of possible actions from 'state'.
        """
        return [item[1] for item in self._possibleActions.keys() if item[0] == state]


    def getTransitionStatesAndProbs(self, state, action):
        """
        Returns list of (nextState, prob) pairs
        representing the states reachable
        from 'state' by taking 'action' along
        with their transition probabilities.

        Note that in Q-Learning and reinforcment
        learning in general, we do not know these
        probabilities nor do we directly model them.
        """
        return [(self._possibleActions[(state, action)], 1), ]

    def getReward(self, state, action, nextState):
        """
        Get the reward for the state, action, nextState transition.

        Not available in reinforcement learning.
        """
        return self._rewards[nextState]


    def isTerminal(self, state):
        """
        Returns true if the current state is a terminal state.  By convention,
        a terminal state has zero future rewards.  Sometimes the terminal state(s)
        may have no possible actions.  It is also common to think of the terminal
        state as having a self-loop action 'pass' with zero reward; the formulations
        are equivalent.
        """
        return False

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########


class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
  def _getGrid(self, x_min, y_min, x_max, y_max):
      x_min = int(max(1, x_min))
      x_max = int(min(32, x_max))
      y_min = int(max(1, y_min))
      y_max = int(min(self._maxy, y_max))

      all_cells = set()
      for x in range(x_min, x_max + 1):
          for y in range(y_min, y_max + 1):
              all_cells.add((x, y))
      return all_cells.difference(self.walls)


  def printLegalActions(self, gameState, agent_idx):
      state = gameState.getAgentState(agent_idx)

      print "My index: ", agent_idx
      print "My position:", state.getPosition()
      print "ScaredTimer / NumCarrying / NumReturned", state.scaredTimer, state.numCarrying, state.numReturned, \
          state.isPacman, state.start

      try:
          legal_actions = gameState.getLegalActions(agent_idx)
          print 'legal actions:', legal_actions
          for action in legal_actions:
              try:
                  successor = self.getSuccessor(gameState, action)
                  succ_state = successor.getAgentState(agent_idx)
                  print "Position after the action", action, succ_state.getPosition(), succ_state.getDirection(), \
                      succ_state.scaredTimer, succ_state.numCarrying, succ_state.numReturned
              except Exception:
                  pass
      except AttributeError:
          pass

  def getDistanceHome(self, pos):
      x, _ = pos
      if (self.homeBoundaryCells[0][0] - x) * self.sign > 0:
          return 0
      distances = [self.distancer.getDistance(pos, cell) for cell in self.homeBoundaryCells]
      return min(distances)
 
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.enemy_indices = self.getOpponents(gameState)
    self.walls = set(gameState.data.layout.walls.asList())
    self._maxy = max([item[1] for item in self.walls])
    self.sign = 1 if gameState.isOnRedTeam else -1

    homeXBoundary = self.start[0] + 15 * self.sign
    cells = [(homeXBoundary, y) for y in range(1, self._maxy)]
    self.homeBoundaryCells = [item for item in cells if item not in self.walls]


    print "Walls:", self.walls
    # Agent info
    enemy_capsules = self.getCapsules(gameState)
    print "Enemy's Capsules: ", enemy_capsules
    print "Enemy's indices:", self.enemy_indices
    print "Enemy's food:", self.getFood(gameState).asList()


    team_indices = self.getTeam(gameState)
    print "Start Position", self.start
    print "Our indices", team_indices
    print "Our Capsules:", self.getCapsulesYouAreDefending(gameState)
    print "Our Food:", self.getFoodYouAreDefending(gameState).asList()

    #print "Distances:", self.getMazeDistance((3, 3), (1, 1))


  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    vicinity = 5
    food_base_reward = 0.1
    #distance_home_reward = 0.01
    trap_reward = -1
    enemy_position_reward = -1
    go_home_reward = 1


    #legal_actions = gameState.getLegalActions(self.index)
    #self.printLegalActions(gameState, self.index)
    myState = gameState.getAgentState(self.index)
    myPos = myState.getPosition()
    distance_home = self.getDistanceHome(myPos)
    x, y = myPos
    myPos = (int(x), int(y))

    # Generate GRID
    grid = self._getGrid(x - vicinity, y - vicinity, x + vicinity, y + vicinity)
    grid = {cell for cell in grid if self.distancer.getDistance(myPos, cell) <= vicinity}
    mdp = PacmanMDP(myPos, grid)


    # Positive rewards for food
    foodPositions = self.getFood(gameState).asList()
    foodLeft = len(foodPositions)
    for food in foodPositions:
        for cell in grid:
            distance = self.distancer.getDistance(cell, food)
            mdp.addReward(cell, food_base_reward/max(distance, .5))


    # Rewards for ghosts in vicinity
    for idx in self.getOpponents(gameState):
        enemyState = gameState.getAgentState(idx)
        enemyPos = enemyState.getPosition()
        if not enemyPos:
            continue

        # No rewards if enemy on our territory
        if enemyState.isPacman:
            continue

        # No rewards if we are ghost
        #if not myState.isPacman:
        #    continue

        # Negative reward for enemy's positions nearby
        enemy_distance = self.distancer.getDistance(myPos, enemyPos)
        reward = enemy_position_reward * foodLeft / 2 * (vicinity - enemy_distance + 1)
        #mdp.addRewardWithNeighbours(enemyPos, reward)
        mdp.addReward(enemyPos, reward)

        # Negative reward for going in trapping positions
        for cell in grid:
            cell_to_home_distance = self.getDistanceHome(cell)
            if cell_to_home_distance > distance_home:
                reward = (cell_to_home_distance - distance_home) * trap_reward
                mdp.addReward(cell, reward)


    # Rewards for going home when we carry enough food or game is close to an end
    timeLeft = gameState.data.timeleft // 4
    if (foodLeft <= 2) or (timeLeft < 30):
        print "Food/time left", foodLeft, timeLeft
        for cell in grid:
            distance = self.getDistanceHome(cell)
            if distance < distance_home:
                reward = go_home_reward/max(distance, .5)
                #print "Reward for going home:", reward
                mdp.addReward(cell, reward)




    #mdp.print_state()
    #print grid

    evaluator = ValueIterationAgent(mdp, iterations=50)
    return evaluator.getAction(myPos)

    #print x, y, grid

    # Boundaries
    startX = self.start[0]
    homeBoundaryX = min(17, startX + 15)
    #enemyBoundaryX = max(17, startX + 16, startX - 16)
    #print(homeBoundaryX)#, enemyBoundaryX)

    for idx in self.enemy_indices:
        #self.printLegalActions(gameState, idx)
        pass


    # Additional info
    #print 'Scared Timer:', gameState.data.agentStates[self.index].scaredTimer
    #print 'Time Left:', gameState.data.timeleft // 4
    #print self.final(gameState)

    '''
    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction
      '''

    #return random.choice(bestActions)
    legal_actions = gameState.getLegalActions(self.index)
    return random.choice(legal_actions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

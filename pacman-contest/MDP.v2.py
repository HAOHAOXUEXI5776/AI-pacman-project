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
            #next_state, _ = self.mdp.getTransitionStatesAndProbs(state, action)[0]
            #action_values[action] = self.values[next_state]
            #print state, next_state, self.values[next_state]
        return action_values.argMax()


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)


    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


    def print_state(self):
        for state in self.values.keys():
            qvalue = self.values[state]
            reward = self.mdp.getReward(state, '', state)
            print 'state/qvalue/reward', state, qvalue, reward


class PacmanMDP():
    def __init__(self, startState, states):
        self._startState = startState
        self._states = states

        legalMoves = util.Counter()
        for state in states:
            x, y = state
            #legalMoves[(state, Directions.STOP)] = state
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

    '''
    def print_state(self):
        print "States", self._states
        print "Rewards", self._rewards
        print "Possible Actions", self._possibleActions
    '''

    def getScaledStateReward(self, state):
        rewards = [abs(self._rewards[key]) for key in self._rewards.keys()]
        maxReward = max(rewards)
        return self._rewards[state]/maxReward

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
        if action == Directions.STOP:
            return -10.
        return min(self._rewards[state], self._rewards[nextState])


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
      x_max = int(min(self._maxx, x_max))
      y_min = int(max(1, y_min))
      y_max = int(min(self._maxy, y_max))

      all_cells = set()
      for x in range(x_min, x_max + 1):
          for y in range(y_min, y_max + 1):
              all_cells.add((x, y))
      return all_cells.difference(self.walls)


  def getDistanceHome(self, pos):
      x, _ = pos
      #print self.sign, x, self.homeBoundaryCells
      if (self.homeBoundaryCells[0][0] - x) * self.sign > 0:
          return 0
      distances = [self.distancer.getDistance(pos, cell) for cell in self.homeBoundaryCells]
      return min(distances)
 
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.warnings = 0

    # Enemy and teammate indices
    self.enemy_indices = self.getOpponents(gameState)
    team_indices = self.getTeam(gameState)
    team_indices.remove(self.index)
    self.teammate_index = team_indices[0]

    # Analysing layout
    self.walls = set(gameState.data.layout.walls.asList())
    self._maxx = max([item[0] for item in self.walls])
    self._maxy = max([item[1] for item in self.walls])
    self.sign = 1 if gameState.isOnRedTeam(self.index) else -1

    # Determining home boundary
    self.homeXBoundary = self.start[0] + ((self._maxx // 2 - 1) * self.sign)
    cells = [(self.homeXBoundary, y) for y in range(1, self._maxy)]
    self.homeBoundaryCells = [item for item in cells if item not in self.walls]

    # Determining legal actions count for all cells
    valid_cells = self._getGrid(1, 1, self._maxx, self._maxy)
    self._legalActions = util.Counter()
    for cell in valid_cells:
        x, y = cell
        if (x - 1, y) in valid_cells:
            self._legalActions[cell] += 1
        if (x + 1, y) in valid_cells:
            self._legalActions[cell] += 1
        if (x, y - 1) in valid_cells:
            self._legalActions[cell] += 1
        if (x, y + 1) in valid_cells:
            self._legalActions[cell] += 1

    '''
    print "Home boundary cells: ", self.homeBoundaryCells


    print "Walls:", self.walls
    # Agent info
    enemy_capsules = self.getCapsules(gameState)
    print "Enemy's Capsules: ", enemy_capsules
    print "Enemy's indices:", self.enemy_indices
    print "Enemy's food:", self.getFood(gameState).asList()

    print "Start Position", self.start
    print "Our indices", team_indices
    print "Our Capsules:", self.getCapsulesYouAreDefending(gameState)
    print "Our Food:", self.getFoodYouAreDefending(gameState).asList()
    '''

  def isHomeArena(self, cell):
      x, _ = cell
      x1 = self.start[0]
      x2 = self.homeXBoundary
      return x1 <= x <= x2 or x2 <= x <= x1

  def assignRewards(self, grid, mdp, rewardShape, myPos, targetPos):
      rewards = []
      distanceToTarget = self.distancer.getDistance(targetPos, myPos)
      for cell in grid:
          distance = self.distancer.getDistance(cell, targetPos)
          if distance <= distanceToTarget:
              reward = rewardShape / max(float(distance), .5)
              maxNoise = reward / 5.
              reward += random.uniform(-maxNoise, maxNoise)
              mdp.addReward(cell, reward)
              rewards.append((myPos, cell, distance, reward))
      return rewards

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    start = time.time()

    vicinity = 6
    goHomeReward = 1
    foodRwdShape = 0.1

    trapRwdShape = -0.1
    ghostAttackRwdShape = -1.5
    ppRwdShape = 1

    myState = gameState.getAgentState(self.index)
    myPos = myState.getPosition()
    distance_home = self.getDistanceHome(myPos)
    x, y = myPos
    #myPos = (int(x), int(y))

    teammateState = gameState.getAgentState(self.teammate_index)
    teammatePos = teammateState.getPosition()

    # Generate GRID
    grid = self._getGrid(x - vicinity, y - vicinity, x + vicinity, y + vicinity)
    grid = {cell for cell in grid if self.distancer.getDistance(myPos, cell) <= vicinity}
    mdp = PacmanMDP(myPos, grid)


    # Positive rewards for food
    foodPositions = self.getFood(gameState).asList()
    foodLeft = len(foodPositions)
    if foodLeft > 2:
        for foodPos in foodPositions:
            self.assignRewards(grid, mdp, rewardShape=foodRwdShape, myPos=myPos,
                targetPos=foodPos)


    # Rewards for ghosts in vicinity
    enemyNearby = False
    enemies = []
    for idx in self.getOpponents(gameState):
        enemyState = gameState.getAgentState(idx)
        enemyPos = enemyState.getPosition()

        if enemyPos:
            enemy_distance = self.distancer.getDistance(myPos, enemyPos)
            if enemy_distance <= 5:
                enemyNearby = True
                enemies.append((enemyState, enemyPos))

    if enemyNearby:
        # No rewards if enemy on our territory
        if enemyState.isPacman:
            pass

        # Negative reward for enemy's positions nearby
        enemyMinDistance = 5
        enemyScaredTimer = min([item.scaredTimer for item, _ in enemies])
        for enemyState, enemyPos in enemies:
            if enemyState.scaredTimer > 2:
                continue
            enemy_distance = self.distancer.getDistance(myPos, enemyPos)
            reward = ghostAttackRwdShape * foodLeft * (vicinity - enemy_distance + 1.)
            #print "Reward for enemy nearby, EnemyPos / distance / reward", enemyPos, enemy_distance, reward
            mdp.addRewardWithNeighbours(enemyPos, reward)
            enemyMinDistance = min(enemyMinDistance, enemy_distance)


        for cell in grid:
            # No rewards for cells in home arena
            if self.isHomeArena(cell) or enemyScaredTimer > 9:
                continue

            # Negative reward for going in trapping positions
            cell_to_home_distance = self.getDistanceHome(cell)
            #print "Distance home / cell distance home", distance_home, cell_to_home_distance
            enemyDistCoef = float(6 - enemyMinDistance) / 2.
            if cell_to_home_distance > distance_home:
                reward = float(cell_to_home_distance - distance_home) * trapRwdShape * enemyDistCoef
                mdp.addReward(cell, reward)

            # Negative rewards for going in cells with 1 legal action
            legalActions = self._legalActions[cell]
            if legalActions == 1:
                reward = float(trapRwdShape * enemyDistCoef * 2)
                mdp.addRewardWithNeighbours(cell, reward)
            if legalActions == 2 and enemyMinDistance <=3:
                mdp.addRewardWithNeighbours(cell, trapRwdShape)

            # Positive reward for going towards Power pellets
            for pelletPos in self.getCapsules(gameState):
                self.assignRewards(grid, mdp, rewardShape=ppRwdShape,
                    myPos=myPos, targetPos=pelletPos)

        # Positive rewards for bringing food home
        foodCarriyng = min(myState.numCarrying, 10)
        rewardShape = goHomeReward * max(foodCarriyng - 1, 0) / 10
        self.assignGoHomeRewards(grid, mdp, rewardShape, myPos)


    # Rewards for going home when we carry enough food or game is close to an end
    timeLeft = gameState.data.timeleft // 4
    goingHome = (foodLeft <= 2) or (timeLeft < 40)
    if goingHome:
        self.assignGoHomeRewards(grid, mdp, goHomeReward, myPos)
        #print "Food/time left", foodLeft, timeLeft
        #for targetCell in self.homeBoundaryCells:
        #    rewards = self.assignRewards(grid, mdp, rewardShape=goHomeReward, myPos=myPos, targetPos=targetCell)
        #print 'MiddlePos/myPos/Rewards', targetCell, myPos, rewards

    if myState.numCarrying > 5 and self.getDistanceHome(myPos) <= 3:
        self.assignGoHomeRewards(grid, mdp, goHomeReward, myPos)


    # Rewards to be close to teammate pacman

    rewardShape = 250. / (250 + self.distancer.getDistance(myPos, self.start)) - 1
    #print rewardShape
    self.assignRewards(grid, mdp, rewardShape=rewardShape, myPos=myPos, targetPos=teammatePos)
    mdp.addRewardWithNeighbours(teammatePos, rewardShape)

    #reward = -1 / max(1, self.distancer.getDistance(myPos, teammatePos))
    #mdp.addRewardWithNeighbours(teammatePos, reward)


    if x > 1:
        # if goingHome:
        # if ghostNearby:
        self.debugClear()
        for state in mdp.getStates():
            reward = mdp.getScaledStateReward(state)
            if reward > 0:
                color = [0, max(1, reward), 0]
            else:
                color = [max(0, abs(reward)), 0, 0]
            self.debugDraw([state], color)

    evaluator = ValueIterationAgent(mdp, discount=0.8, iterations=100)
    #if pacmanIsUnderTheThreat:
    #    evaluator.print_state()

    bestAction = evaluator.getAction(myPos)

    timeElapsed = time.time() - start
    if timeElapsed > 0.5:
        self.warnings += 1
    if goingHome:
        #print "myPos/myIndex/chosenAction/timeConsumed", myPos, self.index, bestAction, time.time() - start
        #evaluator.print_state()
        pass
    #print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    return bestAction


  def final(self, gameState):
      #print "Warnings count:", self.warnings
      CaptureAgent.final(self, gameState)

  def assignGoHomeRewards(self, grid, mdp, rewardShape, myPos):
      for targetCell in self.homeBoundaryCells:
          rewards = self.assignRewards(grid, mdp, rewardShape=rewardShape, myPos=myPos, targetPos=targetCell)
      return rewards

  '''
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
  '''

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
    return None

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

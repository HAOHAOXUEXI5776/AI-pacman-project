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
import numpy as np


class ValueIterationAgent():
    """
        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount, iterations):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0

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
        # print(state, action)
        qvalue = 0
        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, next_state)
            # print prob, reward, self.discount
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
            # next_state, _ = self.mdp.getTransitionStatesAndProbs(state, action)[0]
            # action_values[action] = self.values[next_state]
            # print state, next_state, self.values[next_state]
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


class MDPGrid():
    def __init__(self, startState, vicinity):
        self.vicinity = vicinity
        self._startState = startState
        self._states = set()
        self._states.add(startState)
        self._boundaryStates = set()

    def getStartState(self):
        """
        Return the start state of the MDP.
        """
        return self._startState

    def getStates(self):
        """
        Return a list of all states in the MDP.
        Not generally possible for large MDPs.
        """
        return list(self._states)

    def getBoundaryStates(self):
        return list(self._boundaryStates)

    def addState(self, state):
        self._states.add(state)

    def setBoundaryState(self, state):
        self._boundaryStates.add(state)


class PacmanMDP():
    def __init__(self, grid):
        self.grid = grid
        states = grid.getStates()

        # Deraive legal actions inside of given grid
        legalMoves = util.Counter()
        for state in states:
            x, y = state
            # legalMoves[(state, Directions.STOP)] = state
            if (x - 1, y) in states:
                legalMoves[(state, Directions.WEST)] = (x - 1, y)
            if (x + 1, y) in states:
                legalMoves[(state, Directions.EAST)] = (x + 1, y)
            if (x, y - 1) in states:
                legalMoves[(state, Directions.SOUTH)] = (x, y - 1)
            if (x, y + 1) in states:
                legalMoves[(state, Directions.NORTH)] = (x, y + 1)
        self._possibleActions = legalMoves
        self._rewards = util.Counter()

    def addReward(self, state, reward):
        maxNoise = reward / 5.
        reward += random.uniform(-maxNoise, maxNoise)
        self._rewards[state] += reward

    def addRewardWithNeighbours(self, state, reward):
        x, y = state
        self._rewards[state] += reward
        self._rewards[(x - 1, y)] += reward / 2
        self._rewards[(x + 1, y)] += reward / 2
        self._rewards[(x, y - 1)] += reward / 2
        self._rewards[(x, y + 1)] += reward / 2

    def getStates(self):
        """
        Return a list of all states in the MDP.
        Not generally possible for large MDPs.
        """
        #return list(self._states)
        return self.grid.getStates()

    def getStartState(self):
        """
        Return the start state of the MDP.
        """
        #return self._startState
        return self.grid.getStartState()


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
            return 0.
        return self._rewards[nextState]
        #return min(self._rewards[state], self._rewards[nextState])


    def getScaledStateReward(self, state):
        rewards = [abs(self._rewards[key]) for key in self._rewards.keys()]
        maxReward = max(rewards)
        return self._rewards[state]/maxReward

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
        first='OffensiveReflexAgent', second='DefensiveReflexAgent'):
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
        valid_cells = self.getGrid(1, 1, self._maxx, self._maxy)
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

        # Position history
        self._positionsHistory = []

    def getGrid(self, x_min, y_min, x_max, y_max):
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
        # print self.sign, x, self.homeBoundaryCells
        if (self.homeBoundaryCells[0][0] - x) * self.sign > 0:
            return 0
        distances = [self.distancer.getDistance(pos, cell) for cell in self.homeBoundaryCells]
        return min(distances)

    def isHomeArena(self, cell):
        x, _ = cell
        x1 = self.start[0]
        x2 = self.homeXBoundary
        return x1 <= x <= x2 or x2 <= x <= x1


    def getChildrenStates(self, state):
        childrenStates = []
        x, y = state
        if (x - 1, y) not in self.walls:
            childrenStates.append((x - 1, y))
        if (x + 1, y) not in self.walls:
            childrenStates.append((x + 1, y))
        if (x, y - 1) not in self.walls:
            childrenStates.append((x, y - 1))
        if (x, y + 1) not in self.walls:
            childrenStates.append((x, y + 1))
        return childrenStates

    def expandToPacmanGrid(self, grid, currentPos, vicinity):
        startState = grid.getStartState()
        baseDistance = self.distancer.getDistance(currentPos, startState)
        children = self.getChildrenStates(currentPos)
        distances = [self.distancer.getDistance(state, grid.getStartState()) for state in children]
        for idx, distance in enumerate(distances):
            newState = children[idx]
            if distance > baseDistance and distance <= vicinity:
                grid.addState(newState)
                self.expandToPacmanGrid(grid, newState, vicinity)
            if distance == vicinity + 1:
                grid.setBoundaryState(currentPos)

    def assignRewards(self, grid, mdp, rewardShape, myPos, targetPos):
        rewards = []
        distanceToTarget = self.distancer.getDistance(targetPos, myPos)
        for cell in grid.getStates():
            distance = self.distancer.getDistance(cell, targetPos)
            if distance <= distanceToTarget:
                reward = rewardShape / max(float(distance), .5)
                maxNoise = reward / 5.
                reward += random.uniform(-maxNoise, maxNoise)
                mdp.addReward(cell, reward)
                rewards.append((myPos, cell, distance, reward))
        return rewards


    def assignFoodRewards(self, gameState, myPos, mdp, foodRwdShape):
        vicinityCells = mdp.grid.getStates()

        # Positive rewards for food
        foodPositions = {item for item in self.getFood(gameState).asList()}
        foodLeft = len(foodPositions)
        visibleFood = {pos for pos in foodPositions if pos in vicinityCells}
        notVisibleFood = foodPositions.difference(visibleFood)

        if foodLeft > 2:
            # Assing rewards for food in current grid
            for foodPos in visibleFood:
                distance = self.distancer.getDistance(myPos, foodPos)
                reward = foodRwdShape / distance
                mdp.addReward(foodPos, reward)

            # Assign rewards for food outside of visible grid
            for cell in mdp.grid.getBoundaryStates():
                distances = np.array([self.distancer.getDistance(cell, foodPos) for foodPos in notVisibleFood])
                foodNum = 6
                indices = np.argsort(distances)[:foodNum]
                distance_sum = distances[indices].sum() + mdp.grid.vicinity * indices.shape[0]

                reward = foodNum * foodRwdShape / max(1., distance_sum)
                mdp.addReward(cell, reward)

        return visibleFood, foodLeft


    def assignPowerPelletRewards(self, gameState, mdp, rwdShape):
        vicinityCells = mdp.getStates()
        for pelletPos in self.getCapsules(gameState):
            if pelletPos in vicinityCells:
                mdp.addReward(pelletPos, rwdShape)
            else:
                for cell in mdp.grid.getBoundaryStates():
                    distance = self.distancer.getDistance(cell, pelletPos)
                    reward = rwdShape / distance
                    mdp.addReward(cell, reward)




    def assignGoHomeRewads(self, mdp, rwdShape):
        for cell in self.homeBoundaryCells:
            if cell in mdp.grid.getStates():
                mdp.addReward(cell, rwdShape)
        for cell in mdp.grid.getBoundaryStates():
            distance = self.getDistanceHome(cell)
            if distance > 0:
                mdp.addReward(cell, rwdShape / (distance / 2.))

    def assignGhostRewards(self, mdp, ghostPos, reward):
        x, y = ghostPos
        if not self.isHomeArena(ghostPos):
            mdp.addReward(ghostPos, reward)
        for pos in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
            if pos not in mdp.getStates():
                continue
            if not self.isHomeArena(pos):
                mdp.addReward(pos, reward/2.)


    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        self.debugClear()
        start = time.time()
        vicinity = 6
        niterations = 20
        gamma = 0.9

        goHomeReward = 1.
        foodRwdShape = 0.1
        teammateRwdShape = -0.15

        trapRwdShape = -0.3
        ghostAttackingRwdShape = -3
        ppRwdShape = 1.


        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        self._positionsHistory.append(myPos)
        distance_home = self.getDistanceHome(myPos)
        x, y = myPos

        # Teammate
        teammateState = gameState.getAgentState(self.teammate_index)
        teammatePos = teammateState.getPosition()

        # Generate GRID
        grid = MDPGrid(myPos, vicinity)
        self.expandToPacmanGrid(grid, myPos, vicinity)
        mdp = PacmanMDP(grid)


        # Assign positive rewards for food
        visibleFood, foodLeft = self.assignFoodRewards(gameState, myPos, mdp, foodRwdShape)


        # Determinng if there is enemies nearby
        ghostNearby = False
        underThreat = False
        timeToThreat = 100
        enemies = []
        for idx in self.getOpponents(gameState):
            enemyState = gameState.getAgentState(idx)
            enemyPos = enemyState.getPosition()

            if enemyPos and not enemyState.isPacman:
                enemy_distance = self.distancer.getDistance(myPos, enemyPos)
                if enemy_distance <= 5:
                    ghostNearby = True
                    enemies.append((enemyState, enemyPos))

                    if enemyState.scaredTimer < 10:
                        underThreat = True
                        timeToThreat = min(timeToThreat, enemyState.scaredTimer)

        #pass

        if ghostNearby:
            if underThreat:
                # Positive rewards for bringing food home
                foodCarriyng = min(myState.numCarrying, 10)
                rewardShape = goHomeReward * foodCarriyng / 20
                self.assignGoHomeRewads(mdp, rewardShape)

            if underThreat:
                # Assign positive rewards for Power Pellets
                self.assignPowerPelletRewards(gameState, mdp, ppRwdShape)

                # Assign rewards for going into trap positions
                for cell in mdp.getStates():
                    # Negative rewards for going in cells with 1 legal action
                    legalActions = self._legalActions[cell]
                    if legalActions == 1:
                        #enemyDistCoef = (6 - enemyMinDistance) / 2.
                        #reward = float(trapRwdShape)
                        mdp.addReward(cell, trapRwdShape)

                    if cell in visibleFood:
                        mdp.addReward(cell, -foodRwdShape)

                for enemyState, enemyPos in enemies:
                    scaredCoefficient = 1./ max(1., enemyState.scaredTimer - 3.)
                    if enemyState.scaredTimer < 10:
                        reward = ghostAttackingRwdShape * scaredCoefficient
                        self.assignGhostRewards(mdp, enemyPos, reward)

                    '''
                    # Negative reward for going in trapping positions
                    cellToHomeDistance = self.getDistanceHome(cell)
                    
                    foodCoefficient = max(foodCarriyng, 1) / 5.
                    dist = max(cellToHomeDistance - distance_home, 0)
                    reward = dist * enemyDistCoef * trapRwdShape * foodCoefficient
                    mdp.addReward(cell, reward)
                    '''

            if not underThreat:
                # Assign negative rewards for Power Pellets
                self.assignPowerPelletRewards(gameState, mdp, -ppRwdShape / 10.)

            # Negative reward for enemy's positions nearby
            enemyMinDistance = 5
            # enemyScaredTimer = min([item.scaredTimer for item, _ in enemies])
            for enemyState, enemyPos in enemies:
                if enemyState.scaredTimer > 2:
                    continue
                if self.isHomeArena(enemyPos):
                    continue
                enemy_distance = self.distancer.getDistance(myPos, enemyPos)
                reward = ghostAttackingRwdShape * (5. - enemy_distance + 1.)
                mdp.addRewardWithNeighbours(enemyPos, reward)
                enemyMinDistance = min(enemyMinDistance, enemy_distance)






        # Rewards to be close to teammate pacman
        if (teammatePos[0] == self.homeXBoundary) or (not self.isHomeArena(teammatePos)):
            mdp.addReward(teammatePos, teammateRwdShape)

        # Rewards for going home when we carry enough food or game is close to an end
        timeLeft = gameState.data.timeleft // 4
        goingHome = (foodLeft <= 2) or (timeLeft < (self.getDistanceHome(myPos) + 5)) or (timeLeft < 40)
        if goingHome:
            self.assignGoHomeRewads(mdp, goHomeReward)


        evaluator = ValueIterationAgent(mdp, discount=gamma, iterations=niterations)
        # if pacmanIsUnderTheThreat:
        #    evaluator.print_state()


        if x > 28:
        #if goingHome:
        #if ghostNearby:
            self.debugClear()
            for state in grid.getStates():
                reward = mdp.getScaledStateReward(state)
                if reward > 0:
                    color = [0, max(1, reward), 0]
                else:
                    color = [max(0, abs(reward)), 0, 0]
                self.debugDraw([state], color)

        bestAction = evaluator.getAction(myPos)
        #print 'myPos', myPos
        #evaluator.print_state()

        timeElapsed = time.time() - start
        if timeElapsed > 0.5:
            self.warnings += 1
        #if goingHome:
            # print "myPos/myIndex/chosenAction/timeConsumed", myPos, self.index, bestAction, time.time() - start
            # evaluator.print_state()
        #    pass

        return bestAction

    def final(self, gameState):
        # print "Warnings count:", self.warnings
        CaptureAgent.final(self, gameState)

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

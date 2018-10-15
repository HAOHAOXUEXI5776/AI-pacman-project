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
import util
from captureAgents import CaptureAgent
from game import Actions
from game import Directions
from util import nearestPoint

#### GLOBAL VARIABLES NEEDED FOR ASTAR ###

GENERIC = False
DEBUG = False
validNextPositions = {}
Walls = set()
NoWalls = set()
nearestEnemyLocation = None
POWER_PELLET_VICINITY = 5
DEFENDING = []
DNUM = 0
nearestEnemyLocation = None
entryPoints = []
latestFoodMissing = None
MODE = 'normal'
DEPTH = 4


###########################################


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
        first='OffensiveReflexAgent', second='DefensiveAstar'):
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

        # Enemy boundary
        y = 1
        x = self.homeXBoundary + 1
        while ((x, y) not in self.walls):
            y += 1
        if self.isHomeArena((x, y)):
            self.enemyXBoundary = x
        else:
            self.enemyXBoundary = self.homeXBoundary - 1
        cells = [(self.enemyXBoundary, y) for y in range(1, self._maxy)]
        self.enemyBoundaryCells = [item for item in cells if item not in self.walls]

        # Enemy Start
        if self.start[0] == 1:
            self.enemyStart = (self._maxx - 1, self._maxy - 1)
        else:
            self.enemyStart = (1, 1)

        distances = np.array([self.distancer.getDistance(cell, self.enemyStart) for cell in self.enemyBoundaryCells])
        indices = np.argsort(distances)[:4]

        self.invasionPoints = []
        for idx in indices:
            self.invasionPoints.append(self.enemyBoundaryCells[idx])

        #print "invasion points", self.invasionPoints

    def shape(self, x):
        return np.exp(-np.sqrt(x) / 4)

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
        if (self.homeBoundaryCells[0][0] - x) * self.sign > 0:
            return 0
        distances = [self.distancer.getDistance(pos, cell) for cell in self.homeBoundaryCells]
        return min(distances)

    def getInvasionDistance(self, pos):
        if not self.isHomeArena(pos):
            return 0
        distances = [self.distancer.getDistance(pos, cell) for cell in self.homeBoundaryCells]
        return min(distances) + 1

    def isHomeArena(self, cell):
        x, _ = cell
        x1 = self.start[0]
        x2 = self.homeXBoundary
        return x1 <= x <= x2 or x2 <= x <= x1


    def getChildrenStates(self, state, walls):
        childrenStates = []
        x, y = state
        if (x - 1, y) not in walls:
            childrenStates.append((x - 1, y))
        if (x + 1, y) not in walls:
            childrenStates.append((x + 1, y))
        if (x, y - 1) not in walls:
            childrenStates.append((x, y - 1))
        if (x, y + 1) not in walls:
            childrenStates.append((x, y + 1))
        return childrenStates

    def expandToPacmanGrid(self, currentPos, vicinity, walls, grid=None):
        if not grid:
            grid = MDPGrid(currentPos, vicinity)
        startState = grid.getStartState()
        baseDistance = self.distancer.getDistance(currentPos, startState)
        children = self.getChildrenStates(currentPos, walls)
        distances = [self.distancer.getDistance(state, grid.getStartState()) for state in children]
        for idx, distance in enumerate(distances):
            newState = children[idx]
            if distance > baseDistance and distance <= vicinity:
                grid.addState(newState)
                self.expandToPacmanGrid(newState, vicinity, walls, grid)
            if distance == vicinity + 1:
                grid.setBoundaryState(currentPos)
        return grid


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
                reward = foodRwdShape * self.shape(distance)
                mdp.addReward(foodPos, reward)

            # Assign rewards for food outside of visible grid
            invadeDistance = self.getInvasionDistance(myPos)
            for cell in mdp.grid.getBoundaryStates():
                if (self.getInvasionDistance(cell) > invadeDistance):
                    continue

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
                    #print "Pellet reward", reward
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
        start = time.time()
        vicinity = 6
        niterations = 100
        gamma = 0.9

        goHomeReward = 10.
        foodRwdShape = 2.
        boundaryRwdShape = .0
        teammateRwdShape = -1.5

        trapRwdShape = -3.
        ghostAttackingRwdShape = -20
        ppRwdShape = 10.


        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        distanceHome = self.getDistanceHome(myPos)
        x, y = myPos

        # Teammate
        teammateState = gameState.getAgentState(self.teammate_index)
        teammatePos = teammateState.getPosition()

        # Generate GRID
        grid = self.expandToPacmanGrid(myPos, vicinity, self.walls)
        mdp = PacmanMDP(grid)

        # Assign positive rewards for food
        visibleFood, foodLeft = self.assignFoodRewards(gameState, myPos, mdp, foodRwdShape)


        # Determining if there is enemies nearby
        ghostNearby = False
        underThreat = False
        timeToThreat = 100
        enemies = []
        for idx in self.getOpponents(gameState):
            enemyState = gameState.getAgentState(idx)
            enemyPos = enemyState.getPosition()

            if enemyPos and not enemyState.isPacman:
                enemies.append((enemyState, enemyPos))
                enemy_distance = self.distancer.getDistance(myPos, enemyPos)
                if enemy_distance <= 5:
                    ghostNearby = True

                    if enemyState.scaredTimer < 10:
                        underThreat = True
                        timeToThreat = min(timeToThreat, enemyState.scaredTimer)

        # Rewards for agent being ghoat and enemy nearby home boundary
        if self.isHomeArena(myPos):
            for cell in self.homeBoundaryCells:
                for enemyState, enemyPos in enemies:
                    if self.distancer.getDistance(cell, enemyPos) <= 2:
                        mdp.addReward(cell, boundaryRwdShape)

                for invasionPos in self.invasionPoints:
                    if self.distancer.getDistance(cell, invasionPos) <= 2:
                        mdp.addReward(cell, boundaryRwdShape)



        # Assigning rewards if for nearby ghosts
        if ghostNearby:

            # Positive rewards for bringing food home
            if underThreat:
                foodCarriyng = min(myState.numCarrying, 10)
                rewardShape = goHomeReward * foodCarriyng / 20
                self.assignGoHomeRewads(mdp, rewardShape)

            # Assign positive rewards for Power Pellets
            if underThreat:
                self.assignPowerPelletRewards(gameState, mdp, ppRwdShape)

            # Assign rewards for going into trap positions
            if underThreat:
                for cell in mdp.getStates():
                    # Negative rewards for going in cells with 1 legal action
                    legalActions = self._legalActions[cell]
                    if legalActions == 1:
                        mdp.addReward(cell, trapRwdShape)

                    #if cell in visibleFood:
                    #    mdp.addReward(cell, -foodRwdShape)


            if underThreat:
                # Negative rewards for enemy positions
                for enemyState, enemyPos in enemies:
                    scaredCoefficient = 1./ max(1., enemyState.scaredTimer - 3.)
                    if enemyState.scaredTimer < 10:
                        reward = ghostAttackingRwdShape * scaredCoefficient
                        self.assignGhostRewards(mdp, enemyPos, reward)

                for cell in mdp.grid.getBoundaryStates():
                    boundaryDistanceHome = self.getDistanceHome(cell)
                    walls = self.walls.union(mdp.getStates())
                    depth = 7
                    expandedArea = self.expandToPacmanGrid(cell, depth, walls)
                    if len(expandedArea.getStates()) < depth * 2:
                        mdp.addReward(cell, trapRwdShape)

                    #if boundaryDistanceHome >= mdp.grid.vicinity + distanceHome:
                    #    mdp.addReward(cell, trapRwdShape)



            if not underThreat:
                # Assign negative rewards for Power Pellets
                self.assignPowerPelletRewards(gameState, mdp, -ppRwdShape / 10.)

                if enemyState.scaredTimer >= 10:
                    mdp.addReward(enemyPos, -ppRwdShape / 10.)


        # Rewards to be close to teammate pacman
        if (teammatePos[0] == self.homeXBoundary) or (not self.isHomeArena(teammatePos)):
            mdp.addReward(teammatePos, teammateRwdShape)


        # Rewards for going home when we carry enough food or game is close to the end
        timeLeft = gameState.data.timeleft // 4
        goingHome = (foodLeft <= 2) or (timeLeft < (self.getDistanceHome(myPos) + 5)) or (timeLeft < 40)
        if goingHome:
            self.assignGoHomeRewads(mdp, goHomeReward)


        evaluator = ValueIterationAgent(mdp, discount=gamma, iterations=niterations)

        #if  14 < x < 18:
        if x < 0:
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


class ReflexCaptureAgentAstar(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """

    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.weights = util.Counter()
        self.discountFactor = 0.65
        self.ValidPos = {}
        self.PrevAction = None
        self.minPelletsToCashIn = 5
        self.maxPelletsToCashIn = 15
        self.AttackHistory = []
        self.DefenceHistory = []
        self.offensiveEntry = None
        self.defensiveEntry = None

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        # Offline data computation, can be utilised further.
        arr = np.zeros((gameState.data.layout.width - 2, gameState.data.layout.height - 2))
        noWallsTemp = set([(index[0][0] + 1, index[0][1] + 1) for index in np.ndenumerate(arr) if
                           not gameState.hasWall(index[0][0] + 1, index[0][1] + 1)])
        WallsTemp = set([(index[0][0] + 1, index[0][1] + 1) for index in np.ndenumerate(arr) if
                         gameState.hasWall(index[0][0] + 1, index[0][1] + 1)])
        # Valid Moves contains all available moves from each available legal state in the map
        for x, y in noWallsTemp:
            availableMoves = []
            if (x + 1, y) not in WallsTemp and 0 < x + 1 < gameState.data.layout.width - 1 and 0 < y < gameState.data.layout.height - 1:
                availableMoves.append((x + 1, y))
            if (x, y + 1) not in WallsTemp and 0 < x < gameState.data.layout.width - 1 and 0 < y + 1 < gameState.data.layout.height - 1:
                availableMoves.append((x, y + 1))
            if (x - 1, y) not in WallsTemp and 0 < x - 1 < gameState.data.layout.width - 1 and 0 < y < gameState.data.layout.height - 1:
                availableMoves.append((x - 1, y))
            if (x, y - 1) not in WallsTemp and 0 < x < gameState.data.layout.width - 1 and 0 < y - 1 < gameState.data.layout.height - 1:
                availableMoves.append((x, y - 1))
            global validNextPositions
            key = str(x) + ',' + str(y)
            validNextPositions[key] = availableMoves
        self.ValidPos = validNextPositions
        global Walls
        Walls = WallsTemp
        global NoWalls
        NoWalls = noWallsTemp
        #########################
        # DEFENSIVE ENTRY POINT #
        #########################
        centralX = (gameState.data.layout.width / 2) - 1
        centralY = (gameState.data.layout.height / 2) - 2
        coordsUpper = []
        coordsLower = []
        coords = []
        for i in range(DEPTH):
            coordsLower.append(
                [location for location in NoWalls if location[0] == (centralX - i) and location[1] <= centralY])
            coordsUpper.append(
                [location for location in NoWalls if location[0] == (centralX - i) and location[1] > centralY])
            coords.append([location for location in NoWalls if location[0] == (centralX - i)])
        if MODE == 'mix':
            self.defensiveEntry = list(set(min(coordsLower, key=len)).union(min(coordsUpper, key=len)))
        if MODE == 'normal':
            self.defensiveEntry = min(coords, key=len)
        global latestFoodMissing
        latestFoodMissing = random.choice(self.getFoodYouAreDefending(gameState).asList())

        #########################
        # OFFENSIVE ENTRY POINT #
        #########################
        centralX = ((gameState.data.layout.width / 2) - 1) + 1
        centralY = (gameState.data.layout.height / 2) - 2
        coordsUpper = []
        coordsLower = []
        coords = []
        for i in range(DEPTH):
            coordsLower.append(
                [location for location in NoWalls if location[0] == (centralX + i) and location[1] <= centralY])
            coordsUpper.append(
                [location for location in NoWalls if location[0] == (centralX + i) and location[1] > centralY])
            coords.append([location for location in NoWalls if location[0] == (centralX + i)])
        if MODE == 'mix':
            self.offensiveEntry = list(set(min(coordsLower, key=len)).union(min(coordsUpper, key=len)))
        if MODE == 'normal':
            self.offensiveEntry = min(coords, key=len)

        if self.index % 2 == 0:
            # Signifies we are the RED Team
            enemyStartState = gameState.getAgentState(1).getPosition()
        else:
            enemyStartState = gameState.getAgentState(0).getPosition()
            swap = self.defensiveEntry
            self.defensiveEntry = self.offensiveEntry
            self.offensiveEntry = swap

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """

        # Append the History of all States played
        self.observationHistory.append(gameState)
        actions = gameState.getLegalActions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        start = time.time()
        values = [self.AproaxQvalue(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())

        bestAction = None
        # Go back to start if there are only 5 food left
        if foodLeft <= 0:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist

        else:
            bestAction = random.choice(bestActions)
        # self.PrevAction = bestAction
        return bestAction

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

    def AproaxQvalue(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def ValueFromQvalue(self, gameState):

        '''
         Given a state this function caliculates the best Q value for the next state i.e, Q(s',a')
        '''

        actions = gameState.getLegalActions(self.index)

        if actions:
            values = [self.AproaxQvalue(gameState, a) for a in actions]
            maxValue = max(values)
            return maxValue
        else:
            return 0

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

    #####################
    # ASTAR LOGIC BEGIN #
    #####################

    def isGhost(self, gameState, index):
        """
        Return the location of the enemy ghost
        """
        pos = gameState.getAgentPosition(index)
        if pos is None:
            return False
        return not (gameState.isOnRedTeam(index) ^ (pos[0] < gameState.getWalls().width / 2))

    def isScared(self, gameState, index):
        """
        check if an agent is in scared state or not
        """
        isScared = bool(gameState.data.agentStates[index].scaredTimer)
        return isScared

    def isPacman(self, gameState, index):
        """
        If we can see the enemy and he is a pacman return true
        """
        pos = gameState.getAgentPosition(index)
        if pos is None:
            return False
        return not (gameState.isOnRedTeam(index) ^ (pos[0] >= gameState.getWalls().width / 2))

    def aStarSearch(self, startPosition, gameState, goalPositions, avoidPositions=[], returnPosition=False):
        """
        This is Astar function which returns the path to the goal positons and avoiding the avoid positions
        """
        walls = gameState.getWalls()

        walls = walls.asList()

        actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        actionVec = [Actions.directionToVector(action) for action in actions]
        actionVec = [tuple(int(number) for number in vector) for vector in actionVec]

        # Each node of the graph is stored as a tuple (Position, path, total cost)_

        currentPos, currentPath, currentCost = startPosition, [], 0


        # we use the priority queue with the implemented huesristic : sum of distances to goals and 100 times distance to avoid positon

        queue = util.PriorityQueueWithFunction(lambda x: x[2] +  # Total cost so far
                                                             (100) * self.getMazeDistance(startPosition, x[0]) if
        x[0] in avoidPositions else 0 +  # Avoid enemy locations
                                        sum([self.getMazeDistance(x[0], Pos) for Pos in
                                             goalPositions]))

        # we put all visited locations in an open list
        visited = set([currentPos])

        while currentPos not in goalPositions:

            possiblePos = [((currentPos[0] + vector[0], currentPos[1] + vector[1]), action) for
                                 vector, action in zip(actionVec, actions)]
            legalPositions = [(position, action) for position, action in possiblePos if position not in walls]

            for position, action in legalPositions:
                if position not in visited:
                    visited.add(position)
                    queue.push((position, currentPath + [action], currentCost + 1))

            if len(queue.heap) == 0:
                return None
            else:
                currentPos, currentPath, currentCost = queue.pop()

        if returnPosition:
            return currentPath, currentPos
        else:
            return currentPath

    def getFoodDistance(self, gameState, action):  #
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        myPos = successor.getAgentState(self.index).getPosition()
        nearestDistance = 0
        if len(foodList) > 0:
            nearestDistance = min(
                [self.getMazeDistance(myPos, food) + abs(self.favoredY - food[1]) for food in foodList])
        return nearestDistance


class DefensiveAstar(ReflexCaptureAgentAstar):
    """
    Note this was Q-value evaluation part borrowed from Baseline purpose of this part was to avoid random actions
    taken by Astar if the path returned is nulll
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

    def attackQvalue(self, gameState, action):
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def computeActionFromQValues(self, state):
        """
          compute the next best action based on the Q values of the baseline Agent
        """
        bestValue = -999999
        bestActions = None
        for action in state.getLegalActions(self.index):
            # for each action in the legal actions get the maximum Q value
            value = self.attackQvalue(state, action)
            if (DEBUG):
                print
                "ACTION: " + action + "           QVALUE: " + str(value)
            if value > bestValue:
                bestActions = [action]
                bestValue = value
            elif value == bestValue:
                bestActions.append(action)
        if bestActions == None:
            return Directions.STOP  # If no legal actions return None
        return random.choice(bestActions)  # Else choose one of the best actions randomly

    def nearestFoodLocation(self, gamestate, ourPosition):
        # Compute distance to the nearest food
        foodList = self.getFoodYouAreDefending(gamestate).asList()
        dists = [(self.getMazeDistance(ourPosition, x), x) for x in foodList]
        if dists:
            return min(dists)[1]

    def chooseAction(self, state):

        hasFooodBeenEatenLastime = False
        foodMissing = []
        foodDefend = self.getFoodYouAreDefending(state).asList()
        if len(self.DefenceHistory):
            prev_state = self.DefenceHistory.pop()

            global nearestEnemyLocation
            if nearestEnemyLocation is None:
                nearestEnemyLocation = state.getInitialAgentPosition(self.getOpponents(state)[0])

            successor = self.getSuccessor(prev_state, self.PrevAction)
            myPos = successor.getAgentState(self.index).getPosition()
            enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
            Ghosts = [agent for agent in enemies if
                      not agent.isPacman and agent.getPosition() is not None and not agent.scaredTimer > 0]
            if successor.getAgentState(self.getOpponents(state)[0]).isPacman or successor.getAgentState(
                    self.getOpponents(state)[1]).isPacman:
                global DEFENDING
                foodMissing = list(set(DEFENDING).difference(foodDefend))
                global latestFoodMissing
                hasFooodBeenEatenLastime = (len(foodDefend) != len(self.getFoodYouAreDefending(prev_state).asList()))
                # print('HasFoodBeenEatenInLastTurn: - - - - -', hasFooodBeenEatenLastime)
                if foodMissing:
                    # print('Latest Missing:', latestFoodMissing)
                    latestFoodMissing = foodMissing[0]
                else:
                    # print('Else Condition: FM:', foodMissing)
                    foodMissing = [latestFoodMissing]

        global DEFENDING
        DEFENDING = foodDefend

        self.DefenceHistory.append(state)

        global validNextPositions
        legalCoordinates = []
        keys = validNextPositions.keys()
        for key in keys:
            possibleMoves = len(validNextPositions[key])
            if possibleMoves == 1:
                x = int(key.split(',')[0])
                y = int(key.split(',')[1])
                legalCoordinates.append((x, y))

        nowalls = []
        for i in range(1, state.data.layout.height):

            if self.index % 2 == 0:
                if state.hasWall(state.data.layout.width / 2 - 1, i) == False:
                    nowalls.append((state.data.layout.width / 2 - 1, i))
            else:
                if state.hasWall(state.data.layout.width / 2, i) == False:
                    nowalls.append((state.data.layout.width / 2, i))

        # Get opponent walls #

        walls = state.getWalls().asList()
        walls = list(set(walls))
        opponentWalls = []
        if self.index % 2 == 0:
            opponentWalls = [w for w in walls if w[0] > 16]  # WARNING: VALUES HARDCODED
        else:
            opponentWalls = [w for w in walls if w[0] < 17]

        avoidPositions = []
        #######################
        # CHOOSE ACTION ASTAR #
        #######################
        food = self.getFood(state)
        enemyIndices = self.getOpponents(state)
        capsules = self.getCapsules(state)

        attackablePacmen = [state.getAgentPosition(i) for i in enemyIndices if
                            self.isPacman(state, i) and self.isGhost(state, self.index)]

        avoidPacmen = [state.getAgentPosition(i) for i in enemyIndices if
                       self.isPacman(state, i) and self.isScared(state, self.index)]

        anyEnemy = [state.getAgentState(i).isPacman for i in enemyIndices]

        nearestFood = self.nearestFoodLocation(state, state.getAgentPosition(self.index))

        if anyEnemy[0] or anyEnemy[1]:

            # if enemy is pacmen and observable attack him
            if attackablePacmen:
                goalPositions = set(attackablePacmen)
            # else go to last food location
            else:
                # print foodMissing
                goalPositions = set(foodMissing)

        else:
            # print foodMissing
            goalPositions = set(foodMissing).union(set(self.defensiveEntry))

        if self.isScared(state, self.index) and (anyEnemy[0] or anyEnemy[1]):

            goalPositions = set(foodMissing)
            avoidPositions = set(avoidPacmen)
            newGoalPositions = []
            newAvoidPositions = []

            # also avoid the corners
            for positions in list(goalPositions):
                pos = str(positions[0]) + ',' + str(positions[1])
                if len(self.ValidPos[pos]) < 2:
                    # Should not go here
                    newAvoidPositions.append(positions)
                else:
                    newGoalPositions.append(positions)
            # if newgoals are zero go to the nearest food
            if len(newGoalPositions) == 0:
                # print 'GOOING TO NEAREST FOOD'
                newGoalPositions.append(nearestFood)
            goalPositions = set(newGoalPositions)
            avoidPositions = set(avoidPositions).union(set(newAvoidPositions))
            # print goalPositions, ' --- ', avoidPositions

        if goalPositions:

            ## if current position is in goal state remove it ##
            currentPos = set([state.getAgentPosition(self.index)])
            if currentPos.issubset(goalPositions):
                goalPositions = goalPositions.difference(currentPos)
            # print goalPositions

            astar_path = self.aStarSearch(state.getAgentPosition(self.index), state, goalPositions, avoidPositions)

        else:
            astar_path = None

        # THIS LOOP BELOW FOR IF GOING BACK IS AN ISSUE IF NO CAPUSLES
        if astar_path:

            # print astar_path, goalPositions, state.getAgentPosition(self.index)
            action_astar = astar_path[0]
            # print 'Iam GOING ASTAR WAY BITCH'
        else:
            # print 'I AM TAKING RANDM SHIT'
            action_astar = self.computeActionFromQValues(state)

        actionToBeExecuted = None
        legalActions = state.getLegalActions(self.index)
        if action_astar in legalActions:
            actionToBeExecuted = action_astar
        else:
            actionToBeExecuted = self.computeActionFromQValues(state)
            # print 'QVALUE CHOICE', actionToBeExecuted

        self.PrevAction = actionToBeExecuted

        return actionToBeExecuted

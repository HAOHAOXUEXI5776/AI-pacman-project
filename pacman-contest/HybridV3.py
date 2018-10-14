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
BELIEF_LOGIC = True
beliefs = []
beliefsInitialized = []
USE_BELIEF_DISTANCE = True
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

    def __init__(self, mdp, discount=0.7, iterations=100):
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
            # print 'state/qvalue/reward', state, qvalue, reward


class PacmanMDP():
    def __init__(self, startState, states):
        self._startState = startState
        self._states = states

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
        # print self.sign, x, self.homeBoundaryCells
        if (self.homeBoundaryCells[0][0] - x) * self.sign > 0:
            return 0
        distances = [self.distancer.getDistance(pos, cell) for cell in self.homeBoundaryCells]
        # print min(distances)
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

        vicinity = 7
        goHomeReward = 1
        foodRwdShape = 0.14

        trapRwdShape = -0.11
        ghostAttackRwdShape = -1.5
        ppRwdShape = 1

        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        distance_home = self.getDistanceHome(myPos)
        x, y = myPos
        # myPos = (int(x), int(y))

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
                # print "Reward for enemy nearby, EnemyPos / distance / reward", enemyPos, enemy_distance, reward
                mdp.addRewardWithNeighbours(enemyPos, reward)
                enemyMinDistance = min(enemyMinDistance, enemy_distance)

            for cell in grid:
                # No rewards for cells in home arena
                if self.isHomeArena(cell) or enemyScaredTimer > 9:
                    continue

                # Negative reward for going in trapping positions
                cell_to_home_distance = self.getDistanceHome(cell)
                # print "Distance home / cell distance home", distance_home, cell_to_home_distance
                enemyDistCoef = float(6 - enemyMinDistance) / 2.
                if cell_to_home_distance > distance_home:
                    reward = float(cell_to_home_distance - distance_home) * trapRwdShape * enemyDistCoef
                    mdp.addReward(cell, reward)

                # Negative rewards for going in cells with 1 legal action
                legalActions = self._legalActions[cell]
                if legalActions == 1:
                    reward = float(trapRwdShape * enemyDistCoef * 2)
                    mdp.addRewardWithNeighbours(cell, reward)
                if legalActions == 2 and enemyMinDistance <= 3:
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
        goingHome = (foodLeft <= 2) or (timeLeft < 40) or (timeLeft < (self.getDistanceHome(myPos) + 10))
        if goingHome:
            self.assignGoHomeRewards(grid, mdp, goHomeReward, myPos)
            # print "Food/time left", foodLeft, timeLeft
            # for targetCell in self.homeBoundaryCells:
            #    rewards = self.assignRewards(grid, mdp, rewardShape=goHomeReward, myPos=myPos, targetPos=targetCell)
            # print 'MiddlePos/myPos/Rewards', targetCell, myPos, rewards

        if myState.numCarrying > 4 and self.getDistanceHome(myPos) < 11:
            self.assignGoHomeRewards(grid, mdp, goHomeReward, myPos)

        # Rewards to be close to teammate pacman
        rewardShape = 250. / (250 + self.distancer.getDistance(myPos, self.start)) - 1
        # print rewardShape
        self.assignRewards(grid, mdp, rewardShape=rewardShape, myPos=myPos, targetPos=teammatePos)
        mdp.addRewardWithNeighbours(teammatePos, rewardShape)

        # reward = -1 / max(1, self.distancer.getDistance(myPos, teammatePos))
        # mdp.addRewardWithNeighbours(teammatePos, reward)

        evaluator = ValueIterationAgent(mdp, discount=0.8, iterations=100)
        # if pacmanIsUnderTheThreat:
        #    evaluator.print_state()

        bestAction = evaluator.getAction(myPos)

        timeElapsed = time.time() - start
        if timeElapsed > 0.5:
            self.warnings += 1
        if goingHome:
            # print "myPos/myIndex/chosenAction/timeConsumed", myPos, self.index, bestAction, time.time() - start
            # evaluator.print_state()
            pass
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        return bestAction

    def final(self, gameState):
        # print "Warnings count:", self.warnings
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


class ReflexCaptureAgentAstar(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """

    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.weights = util.Counter()
        self.isTraining = False
        self.episodesSoFar = 0
        self.epsilon = 0.05
        self.discountFactor = 0.65
        self.ValidPos = {}
        self.alphaLR = 0.0000000002
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
            if (x + 1, y) not in WallsTemp and 0 < x + 1 < 33 and 0 < y < 17:
                availableMoves.append((x + 1, y))
            if (x, y + 1) not in WallsTemp and 0 < x < 33 and 0 < y + 1 < 17:
                availableMoves.append((x, y + 1))
            if (x - 1, y) not in WallsTemp and 0 < x - 1 < 33 and 0 < y < 17:
                availableMoves.append((x - 1, y))
            if (x, y - 1) not in WallsTemp and 0 < x < 33 and 0 < y - 1 < 17:
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

    def getReward(self, gameState):
        foodList = self.getFood(gameState).asList()

        '''
        This is reward function which returns the Cummilative reward in form of a rewward shaping.
        '''
        prev_gameState = self.observationHistory.pop()
        # BRING BACK FOOD TO GET MORE REWARD
        # food current pacman is carrying
        prev_food_carrying = prev_gameState.getAgentState(self.index).numCarrying
        food_carrying = gameState.getAgentState(self.index).numCarrying
        prev_deposited = prev_gameState.getAgentState(self.index).numReturned
        food_deposited = gameState.getAgentState(self.index).numReturned
        food_brought_home = food_deposited - prev_deposited

        net_change_food_carried = food_carrying - prev_food_carrying
        # small reward for eating power capusule

        # small reward for eating food
        if net_change_food_carried > 0:
            eat_reward = 0.2
        else:
            eat_reward = 0

        # small reward for eating capusules
        mypellets_prev = len(self.getCapsules(prev_gameState))
        mypellets_now = len(self.getCapsules(gameState))
        netChangePellets = mypellets_prev - mypellets_now
        if netChangePellets > 0:
            IAtePellete = 1
        else:
            IAtePellete = 0

        # if brought food home give value +10
        if food_brought_home > 0:
            bring_food_value = 20 * food_brought_home
        else:
            bring_food_value = 0

        # REWARD FOR EATING ENEMY PACMAN
        myAgentPosition = prev_gameState.getAgentPosition(self.index)
        eat_enemy_value = 0
        enemy_ate_us_value = 0
        for opponent in self.getOpponents(gameState):
            maybePosition = prev_gameState.getAgentPosition(opponent)
            mayBePositionNow = gameState.getAgentPosition(opponent)
            WasIaPacman = prev_gameState.getAgentState(self.index).isPacman
            AmIaPacman = gameState.getAgentState(self.index).isPacman
            # Enemy is a pacman and he was 1 distance away from us in previous game state

            if maybePosition != None:
                IsEnemyPacman = prev_gameState.getAgentState(opponent).isPacman
                howFarwasEnemy = self.getMazeDistance(myAgentPosition, maybePosition)
                howFarisEnemy = self.getMazeDistance(myAgentPosition, mayBePositionNow)
                if IsEnemyPacman:
                    if howFarwasEnemy < 2:
                        # Enemy has disappeared in the current state means we ate him
                        if howFarisEnemy > 10:
                            eat_enemy_value = 100
                # If I was a pacman and I turned into a ghost and returned to begining
                elif WasIaPacman == True and AmIaPacman == False:
                    myAgentCurrenPos = gameState.getAgentPosition(self.index)
                    if self.start == myAgentCurrenPos:
                        enemy_ate_us_value = -100  # negative reward for being eaten

        # NEGATIVE REWARDS
        ourFoodNow = len(self.getFoodYouAreDefending(gameState).asList())
        ourFoodPrev = len(self.getFoodYouAreDefending(prev_gameState).asList())
        netOurFoodChange = ourFoodPrev - ourFoodNow

        # small negative reward if enemy is eating our food

        if netOurFoodChange > 0:
            enemy_eating_value = -0.2
        else:
            enemy_eating_value = 0

        # small negative reward for enemy eating power pelletes

        pellets_prev = len(self.getCapsulesYouAreDefending(prev_gameState))
        pellets_now = len(self.getCapsulesYouAreDefending(gameState))
        netChangePellets = pellets_prev - pellets_now
        if netChangePellets > 0:
            enemyAtePellete = -1
        else:
            enemyAtePellete = 0

        cummilativeReward = (eat_enemy_value + eat_reward + bring_food_value +
                             enemy_eating_value + enemyAtePellete + enemy_ate_us_value + IAtePellete)

        return cummilativeReward

    def observationFunction(self, gameState):

        '''
         Note this observationFuntion ovverides the function in CaptureAgents

        '''
        if len(self.observationHistory) > 0 and self.isTraining:
            self.update(self.observationHistory.pop(), self.lastAction, gameState, self.getReward(gameState))
            # print self.getReward(gameState)

        return gameState.makeObservation(self.index)

    def update(self, state, action, nextState, reward):

        '''

        This update function updates the weights in the Training phase based on every transition:
        Note: We initial the weights to some values we think are good and then learn them with a slow learning
        rate
        '''

        TD = (reward + self.discountFactor * self.ValueFromQvalue(nextState))
        Qvalue = self.AproaxQvalue(state, action)

        updatedWeights = self.weights.copy()

        FeatureValues = self.getFeatures(state, action)

        for feature in FeatureValues:
            newWeight = updatedWeights[feature] + self.alphaLR * (TD - Qvalue) * FeatureValues[feature]
            updatedWeights[feature] = newWeight
        self.weights = updatedWeights

    def getSetOfMaximumValues(self, counterDictionary):
        return [key for key in counterDictionary.keys() if counterDictionary[key] == max(counterDictionary.values())]

    #####################
    # ASTAR LOGIC BEGIN #
    #####################

    def isGhost(self, gameState, index):
        """
        Returns true ONLY if we can see the agent and it's definitely a ghost
        """
        position = gameState.getAgentPosition(index)
        if position is None:
            return False
        return not (gameState.isOnRedTeam(index) ^ (position[0] < gameState.getWalls().width / 2))

    def isScared(self, gameState, index):
        """
        Says whether or not the given agent is scared
        """
        isScared = bool(gameState.data.agentStates[index].scaredTimer)
        return isScared

    def isPacman(self, gameState, index):
        """
        Returns true ONLY if we can see the agent and it's definitely a pacman
        """
        position = gameState.getAgentPosition(index)
        if position is None:
            return False
        return not (gameState.isOnRedTeam(index) ^ (position[0] >= gameState.getWalls().width / 2))

    def aStarSearch(self, startPosition, gameState, goalPositions, avoidPositions=[], returnPosition=False):
        """
        Finds the distance between the agent with the given index and its nearest goalPosition
        """
        walls = gameState.getWalls()
        width = walls.width
        height = walls.height
        # print width, height
        walls = walls.asList()

        actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        actionVectors = [Actions.directionToVector(action) for action in actions]
        # Change action vectors to integers so they work correctly with indexing
        actionVectors = [tuple(int(number) for number in vector) for vector in actionVectors]

        # Values are stored a 3-tuples, (Position, Path, TotalCost)

        currentPosition, currentPath, currentTotal = startPosition, [], 0
        # Priority queue uses the maze distance between the entered point and its closest goal position to decide which comes first
        queue = util.PriorityQueueWithFunction(lambda entry: entry[2] +  # Total cost so far
                                                             (100) * self.getMazeDistance(startPosition, entry[0]) if
        entry[0] in avoidPositions else 0 +  # Avoid enemy locations
                                        sum([self.getMazeDistance(entry[0], endPosition) for endPosition in
                                             goalPositions]))

        # Keeps track of visited positions
        visited = set([currentPosition])

        while currentPosition not in goalPositions:

            possiblePositions = [((currentPosition[0] + vector[0], currentPosition[1] + vector[1]), action) for
                                 vector, action in zip(actionVectors, actions)]
            legalPositions = [(position, action) for position, action in possiblePositions if position not in walls]

            for position, action in legalPositions:
                if position not in visited:
                    visited.add(position)
                    queue.push((position, currentPath + [action], currentTotal + 1))

            # This shouldn't ever happen...But just in case...
            if len(queue.heap) == 0:
                return None
            else:
                currentPosition, currentPath, currentTotal = queue.pop()

        if returnPosition:
            return currentPath, currentPosition
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

    def attackQvalue(self, gameState, action):
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        bestValue = -999999
        bestActions = None
        for action in state.getLegalActions(self.index):
            # For each action, if that action is the best then
            # update bestValue and update bestActions to be
            # a list containing only that action.
            # If the action is tied for best, then add it to
            # the list of actions with the best value.
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

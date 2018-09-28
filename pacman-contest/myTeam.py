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

import random
import time

import numpy as np

import util
from captureAgents import CaptureAgent
from game import Directions
from util import nearestPoint

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
POWER_PELLET_VICINITY = 3


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

    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.weights = util.Counter()
        self.isTraining = True
        self.episodesSoFar = 0
        self.epsilon = 0.05
        self.discountFactor = 0.75
        self.alphaLR = 0.0000000002


    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.PrevAction = None
        # Offline data computation, can be utilised further.
        arr = np.zeros((32, 16))
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
        global Walls
        Walls = WallsTemp
        global NoWalls
        NoWalls = noWallsTemp
        if self.index % 2 == 0:
            startState = gameState.getAgentState(1).getPosition()
        else:
            startState = gameState.getAgentState(0).getPosition()
        bell = util.Counter().fromkeys(NoWalls, 1)
        bell[startState] = 1
        global beliefsOpponent1
        beliefsOpponent1 = bell
        global beliefsOpponent2
        beliefsOpponent2 = bell
        global OpponentLocation1
        OpponentLocation1 = str(startState[0]) + ',' + str(startState[1])
        global OpponentLocation2
        OpponentLocation2 = str(startState[0]) + ',' + str(startState[1])

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
        print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())

        bestAction = None
        # Go back to start if there are only 5 food left
        if foodLeft <= 5:
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
        self.PrevAction = bestAction
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

        #
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
        Qvalue =  self.AproaxQvalue(state, action)

        updatedWeights = self.weights.copy()

        FeatureValues = self.getFeatures(state, action)

        for feature in FeatureValues:
            newWeight = updatedWeights[feature] + self.alphaLR * (TD-Qvalue) * FeatureValues[feature]
            updatedWeights[feature] = newWeight
        self.weights = updatedWeights

        print 'UPDATED WEIGHTS ARE'
        print self.weights




class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def getFeatures(self, gameState1, action):
        # Making copy of the gamestate
        gameState = gameState1.deepCopy()

        # Initializing Beliefs
        self.observeAllOpponents(gameState)
        features = util.Counter()

        # Our position's successor based on current action:
        successor = self.getSuccessor(gameState, action)
        myPos = successor.getAgentState(self.index).getPosition()
        foodList = self.getFood(successor).asList()
        powerPellets = self.getCapsules(successor)
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        enemyPacmen = [agent for agent in enemies if agent.isPacman and agent.getPosition() is not None]
        Ghosts = [agent for agent in enemies if
                  not agent.isPacman and agent.getPosition() is not None and not agent.scaredTimer > 0]
        scaredGhosts = [agent for agent in enemies if
                        not agent.isPacman and agent.getPosition() is not None and agent.scaredTimer > 0]

        #####################
        # BASELINE FEATURES #
        #####################

        features['successorScore'] = -len(foodList)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(foodList) > 0:  # This should always be True, but better safe than sorry
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance

        ###########################
        # SELF GENERATED FEATURES #
        ###########################

        features['foodLeftToEat'] = len(self.getFood(successor).asList())

        features['ourFoodEaten'] = len(self.getFoodYouAreDefending((successor)).asList())

        features['distanceBetweenOurAgents'] = self.getMazeDistance(successor.getAgentState(
            self.getTeam(gameState)[0]).getPosition(), successor.getAgentState(
            self.getTeam(gameState)[1]).getPosition())

        if GENERIC:
            midway = successor.getAgentState(self.getOpponents(gameState)[1]).start.getPosition()[0] / 2
        else:
            midway = 16
        midwayPoints = [tuple((midway, a)) for a in range(1, midway) if not gameState.hasWall(midway, a)]

        features['minMazeToMiddleFromOurAgent1'] = min(
            [self.getMazeDistance(successor.getAgentState(self.getTeam(gameState)[0]).getPosition(), points) for points
             in midwayPoints])
        features['minMazeToMiddleFromOurAgent2'] = min(
            [self.getMazeDistance(successor.getAgentState(self.getTeam(gameState)[1]).getPosition(), points) for points
             in midwayPoints])

        # Power Pellet Score
        mazeToPowerPellet = 0
        if len(powerPellets) > 0 and len(scaredGhosts) == 0:
            mazeToPowerPellet = min([self.getMazeDistance(myPos, pellet) for pellet in powerPellets])
        features['powerPelletScore'] = max(POWER_PELLET_VICINITY - mazeToPowerPellet, 0)

        # Hunt Enemy
        if len([enemy.isPacman for enemy in enemies]) > 0:
            observableDistance = [self.getMazeDistance(myPos, enemy.getPosition()) for enemy in enemyPacmen]
            # Use the smallest distance
            if len(dists) > 0:
                smallestDist = min(dists)
                return smallestDist

        start = time.time()

        #######################
        # ENEMY APPROXIMATION #
        #######################

        # Cooordinates for the closest enemy ghost that we can observe
        dists = []
        for index in self.getOpponents(successor):
            enemy = successor.getAgentState(index)
            if enemy in Ghosts:
                if USE_BELIEF_DISTANCE:
                    print index, self.getMostLikelyGhostPosition(index)
                    global nearestEnemyLocation
                    nearestEnemyLocation = self.getMostLikelyGhostPosition(index)
                    dists.append(self.getMazeDistance(myPos, self.getMostLikelyGhostPosition(index)))
                else:
                    dists.append(self.getMazeDistance(myPos, enemy.getPosition()))
        features['agent1ToEnemyGhost'] = self.getMazeDistance(
            successor.getAgentState(self.getTeam(gameState)[0]).getPosition(), nearestEnemyLocation)
        features['agent2ToEnemyGhost'] = self.getMazeDistance(
            successor.getAgentState(self.getTeam(gameState)[1]).getPosition(), nearestEnemyLocation)
        features['enemyDistanceToMiddle'] = min(
            [self.getMazeDistance(nearestEnemyLocation, points) for points in midwayPoints])
        return features

    def getWeights(self, gameState, action):
        return {'successorScore': 100, 'distanceToFood': -1}

    ######################
    # BELIEF LOGIC BEGIN #
    ######################
    def getMostLikelyGhostPosition(self, ghostAgentIndex):
        return max(beliefs[ghostAgentIndex])

    def initializeBeliefs(self, gameState):
        beliefs.extend([None for x in range(len(self.getOpponents(gameState)) + len(self.getTeam(gameState)))])
        for opponent in self.getOpponents(gameState):
            self.initializeBelief(opponent, gameState)
        beliefsInitialized.append('done')

    def initializeBelief(self, enemyIndex, gameState):
        belief = util.Counter()
        for gridSpaces in NoWalls:
            belief[gridSpaces] = 1.0
        belief.normalize()
        beliefs[enemyIndex] = belief

    def observeAllOpponents(self, gameState):
        if len(beliefsInitialized):
            for opponent in self.getOpponents(gameState):
                self.observeOneOpponent(gameState, opponent)
        else:
            self.initializeBeliefs(gameState)

    def observeOneOpponent(self, gameState, enemyIndex):
        ourPosition = gameState.getAgentPosition(self.index)
        probabilities = util.Counter()
        maybeIndex = gameState.getAgentPosition(enemyIndex)
        noisyDistance = gameState.getAgentDistances()[enemyIndex]
        if maybeIndex is not None:
            probabilities[maybeIndex] = 1
            beliefs[enemyIndex] = probabilities
            return
        for gridSpaces in NoWalls:
            trueDistance = util.manhattanDistance(gridSpaces, ourPosition)
            modelProb = gameState.getDistanceProb(trueDistance, noisyDistance)
            if modelProb > 0:
                oldProb = beliefs[enemyIndex][gridSpaces]
                probabilities[gridSpaces] = (oldProb + 0.001) * modelProb
            else:
                probabilities[gridSpaces] = 0
        probabilities.normalize()
        beliefs[enemyIndex] = probabilities

    ####################
    # BELIEF LOGIC END #
    ####################

    def getSetOfMaximumValues(self, counterDictionary):
        return [key for key in counterDictionary.keys() if counterDictionary[key] == max(counterDictionary.values())]


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
        invaders = [a for a in enemies if a.isPacman and a.getPosition() is not None]
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

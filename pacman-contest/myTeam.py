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

import numpy as np

import util
from captureAgents import CaptureAgent
from game import Directions
from util import nearestPoint

GENERIC = False
MINIMUM_PROBABILITY = 0.001
beliefs = []
beliefsInitialized = []
Walls = set()
NoWalls = set()
USE_BELIEF_DISTANCE = True


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
        # Offline data computation, can be utilised further.
        arr = np.zeros((32, 16))
        noWallsTemp = set([(index[0][0] + 1, index[0][1] + 1) for index in np.ndenumerate(arr) if
                           not gameState.hasWall(index[0][0] + 1, index[0][1] + 1)])
        WallsTemp = set([(index[0][0] + 1, index[0][1] + 1) for index in np.ndenumerate(arr) if
                         gameState.hasWall(index[0][0] + 1, index[0][1] + 1)])
        # Valid Moves contains all available moves from each available legal state in the map
        validMoves = {}
        walls = set(gameState.getWalls().asList())
        for x, y in noWallsTemp:
            availableMoves = []
            if (x + 1, y) not in WallsTemp:
                availableMoves.append('EAST')
            if (x, y + 1) not in WallsTemp:
                availableMoves.append('NORTH')
            if (x - 1, y) not in WallsTemp:
                availableMoves.append('WEST')
            if (x, y - 1) not in WallsTemp:
                availableMoves.append('SOUTH')
            validMoves[(x, y)] = availableMoves
        global Walls
        Walls = WallsTemp
        global NoWalls
        NoWalls = noWallsTemp

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)

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
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        return random.choice(bestActions)

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
        self.legalPositionsInitialized = False
        # Updating beliefs
        self.observeAllOpponents(gameState)
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        myPos = successor.getAgentState(self.index).getPosition()
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        nonScaredGhosts = [a for a in enemies if not a.isPacman and a.getPosition() is not None]

        features['successorScore'] = -len(foodList)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance

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
        features['minMazeToMidlleFromOurAgent1'] = min(
            [self.getMazeDistance(successor.getAgentState(self.getTeam(gameState)[0]).getPosition(), points) for points
             in midwayPoints])
        features['minMazeToMidlleFromOurAgent2'] = min(
            [self.getMazeDistance(successor.getAgentState(self.getTeam(gameState)[1]).getPosition(), points) for points
             in midwayPoints])

        # Enemy location Approximation
        dists = []
        for index in self.getOpponents(successor):
            enemy = successor.getAgentState(index)
            if enemy in nonScaredGhosts:
                print index, max(beliefs[index])
                dists.append(self.getMazeDistance(myPos, self.getMostLikelyGhostPosition(index)))

        # Use the smallest distance
        if len(dists) > 0:
            smallestDist = min(dists)

            features['ghostDistance'] = smallestDist
        return features

    def getMostLikelyGhostPosition(self, ghostAgentIndex):
        return max(beliefs[ghostAgentIndex])

    def getLegalPositions(self, gameState):
        if not self.legalPositionsInitialized:
            self.legalPositions = []
            walls = gameState.getWalls()
            for x in range(walls.width):
                for y in range(walls.height):
                    if not walls[x][y]:
                        self.legalPositions.append((x, y))
            self.legalPositionsInitialized = True
        return self.legalPositions

    def initializeBeliefs(self, gameState):
        beliefs.extend([None for x in range(len(self.getOpponents(gameState)) + len(self.getTeam(gameState)))])
        for opponent in self.getOpponents(gameState):
            belief = util.Counter()
            for p in (self.getLegalPositions(gameState) if not GENERIC else NoWalls):
                belief[p] = 1.0
            belief.normalize()
            beliefs[opponent] = belief
        beliefsInitialized.append('done')

    def observeAllOpponents(self, gameState):
        if len(beliefsInitialized):
            for opponent in self.getOpponents(gameState):
                self.observeOneOpponent(gameState, opponent)
        else:
            self.initializeBeliefs(gameState)

    def observeOneOpponent(self, gameState, opponentIndex):
        Possible = util.Counter()

        # Our and Enemy Positions
        pacmanPosition = gameState.getAgentPosition(self.index)
        enemyApproximateLocation = gameState.getAgentPosition(opponentIndex)

        # Noisy Distance to the Enemy
        noisyDistance = gameState.getAgentDistances()[opponentIndex]

        # Manhattan Distance if available - Probability is max = 1
        if enemyApproximateLocation is not None:
            Possible[enemyApproximateLocation] = 1
            beliefs[opponentIndex] = Possible
            return

        for p in (self.getLegalPositions(gameState) if not GENERIC else NoWalls):
            # For each legal ghost position, calculate distance to that ghost
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            probability = gameState.getDistanceProb(trueDistance, noisyDistance)

            # Find the probability of getting this noisyDistance if the ghost is at this position
            if probability > 0:
                oldProb = beliefs[opponentIndex][p]
                # Updating Total probability
                Possible[p] = (oldProb + MINIMUM_PROBABILITY) * probability
            else:
                Possible[p] = 0
        Possible.normalize()
        beliefs[opponentIndex] = Possible

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

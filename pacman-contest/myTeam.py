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
import pandas as pd

import util
from captureAgents import CaptureAgent
from game import Directions
from util import nearestPoint

GENERIC = False
MINIMUM_PROBABILITY = 0.001
beliefs = [None, None, None, None]
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
        for x in self.getOpponents(gameState):
            bell = util.Counter().fromkeys(NoWalls, 1)
            beliefs[x] = bell
        # print beliefs

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
        features = util.Counter()

        # Our position's successor based on current action:
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        myPos = successor.getAgentState(self.index).getPosition()
        # print nonScaredGhosts

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

        for enemy in self.getOpponents(gameState):
            print max(beliefs[enemy], key=lambda x: beliefs[enemy][x])
            if gameState.getAgentPosition(enemy) is None:
                currentProbability = self.getProbabilities(gameState, enemy)
                oldProbability = beliefs[enemy]
                updatedProbability = util.Counter()
                for coordinates in oldProbability.keys():
                    update = (oldProbability[coordinates] + 0.001) * currentProbability[coordinates]
                    # print update
                    updatedProbability[coordinates] = update
                updatedProbability.normalize()
                beliefs[enemy] = updatedProbability
            else:
                print gameState.getAgentPosition(enemy)
        # print '*_*_*_*_*_*_*_*_*_*___END_OF_ITERATION__*_*_*_*_*_*_*_*_*_*_*'
        return features

    def getWeights(self, gameState, action):
        return {'successorScore': 100, 'distanceToFood': -1}

    def getProbabilities(self, gameState, opponentIndex):
        Possible = util.Counter()

        # Our Positions
        pacmanPosition = gameState.getAgentPosition(self.index)

        # Noisy Distance to the Enemy
        noisyDistance = gameState.getAgentDistances()[opponentIndex]
        for position in NoWalls:
            trueDistance = util.manhattanDistance(position, pacmanPosition)
            if gameState.getAgentPosition(opponentIndex) is not None:
                probability = 1
                return gameState.getAgentPosition(opponentIndex)
            else:
                probability = gameState.getDistanceProb(trueDistance, noisyDistance)
            Possible[position] = probability
        # Now normalize the probability:
        Possible.normalize()
        return Possible


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

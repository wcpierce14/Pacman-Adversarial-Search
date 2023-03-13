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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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
        newPos = successorGameState.getPacmanPosition() # pacman position after moving
        newFood = successorGameState.getFood() # remaining food
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates] # number of moves that each ghost will remain scared

        numGhosts = len(newGhostStates)
        numFood = successorGameState.getNumFood()
        capsuleCoordinates = currentGameState.getCapsules()

        atFood = False
        atCapsule = False
        oldFood = currentGameState.getFood().asList()
        if newPos in oldFood:
            atFood = True
        
        if newPos in capsuleCoordinates:
            atCapsule = True


        # Calculates distance to closest ghost
        closestGhost = float('inf')
        for ghost in newGhostStates:
            newDist = manhattanDistance(ghost.getPosition(), newPos)
            # If ghost will be at newPos, never go there!
            if newDist == 0:
                return float("-inf")
            if newDist < closestGhost:
                closestGhost = newDist

        # Calculates distance to closest food
        closestFood = float('inf')
        for food in newFood.asList():
            newDist = manhattanDistance(food, newPos)
            if newDist < closestFood:
                closestFood = newDist
        
        # Calculates distance to the closest capsuke
        closestCapsule = float("inf")
        for capsule in capsuleCoordinates:
            newDist = manhattanDistance(capsule, newPos)
            if newDist < closestCapsule:
                closestCapsule = newDist

        # If ghosts are scared, high score == close distance to ghost
        if newScaredTimes != [0 for i in range(numGhosts)]:
            return 1 / closestGhost

        # If ghosts are not scared, maximize closest ghost AND minimize closest food
        else:
            # Large reward for being at food or at capsule
            if atFood or atCapsule:
                if numFood == 0:
                    return 1000
                if len(capsuleCoordinates) == 1:
                    return 1000
                return (closestGhost - closestFood) + 100

            # Return linear combination of distance to closest ghost and distance to closest food
            return 0.75 * closestGhost - closestFood
            

        "*** YOUR CODE HERE ****"

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

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
   
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def minimax(currState, depth, currAgent, alpha, beta):

            # Check terminal state
            if (depth == 0 or currState.isWin() or currState.isLose()):
                return self.evaluationFunction(currState)
    
            # Check the type of the current agent
            currIndex = currAgent

            # If we are at a pacman:
            if currIndex == 0:
                # Get list of pacman actions
                pacmanActions = currState.getLegalActions(0)

                # Loop over actions
                for action in pacmanActions:
                    successor = currState.generateSuccessor(0, action)
                    v2 = minimax(successor, depth - 1, currAgent + 1, alpha, beta) # Calculate minimax value for the first ghost

                    # Check if the value if the minimax successor val is better than alpha
                    if v2 > alpha:
                        alpha = v2

                return alpha
            
            # If we are at a ghost:
            else:
                # Check if it is the last ghost
                if currIndex == (currState.getNumAgents() - 1):

                    # Get list of ghost actions
                    ghostActions = currState.getLegalActions(currIndex)
                    for action in ghostActions:
                        successor = currState.generateSuccessor(currIndex, action)
                        v2 = minimax(successor, depth - 1, 0, alpha, beta) # Calculate the minimax value for the pacman 

                        # Check if the minimax value of the successor is better than beta
                        if v2 < beta:
                            beta = v2

                # We are not at the last ghost in the depth level
                else:
                    # Get list of ghost actions
                    ghostActions = currState.getLegalActions(currIndex)
                    for action in ghostActions:
                        successor = currState.generateSuccessor(currIndex, action)
                        v2 = minimax(successor, depth - 1, currAgent + 1, alpha, beta) # Calculate the minimax value for the next ghost

                        # Check if the minimax value of the successor is better than the beta
                        if v2 < beta:
                            beta = v2
                return beta
            

        # Find the minimax value of the root node and loop through the possible actions to find which yields the optimal action
        totalDepth = self.depth * gameState.getNumAgents()
        rootVal = minimax(gameState, totalDepth, 0, float("-inf"), float("inf")) # minimax value of the root
        for action in gameState.getLegalActions():
            successor = gameState.generateSuccessor(0, action)
            # Check if minimax value of the root equals the minimax value of the sucessor
            if rootVal == minimax(successor, totalDepth - 1, 1, float("-inf"), float("inf")):
                return action



        # util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def minimax(currState, depth, currAgent, alpha, beta):

            # Check terminal state
            if (depth == 0 or currState.isWin() or currState.isLose()):
                return (self.evaluationFunction(currState), None)
    
            # Define currIndex to be the type of agent we are dealing with
            currIndex = currAgent

            # If we are at a pacman:
            if currIndex == 0:
                # Get list of pacman actions
                v = float('-inf')
                pacmanActions = currState.getLegalActions(0)

                # Loop over actions
                for action in pacmanActions:
                    successor = currState.generateSuccessor(0, action) # find the successor state
                    v2 = minimax(successor, depth - 1, currAgent + 1, alpha, beta)[0] # calculate the minimax value of the successor
                    if v2 > v:
                        bestAction = action
                        v = v2

                    # Prune if the successor can force a better outcome with beta
                    if v > beta:
                        return (v, bestAction)

                    # Update alpha if necessary
                    alpha = max(alpha, v)

                return (v, bestAction)
            
            # If we are at a ghost:
            else:
                # Check if it is the last ghost in the depth level
                if currIndex == (currState.getNumAgents() - 1):

                    # Get list of ghost actions
                    v = float('inf')
                    ghostActions = currState.getLegalActions(currIndex)

                    # Loop over all possible ghost actions
                    for action in ghostActions:
                        successor = currState.generateSuccessor(currIndex, action) # compute the successor state
                        v2 = minimax(successor, depth - 1, 0, alpha, beta)[0] # compute the minimax value of the next state

                        if v2 < v:
                            bestAction = action
                            v = v2

                        # Prune if the successor max node can force a better outcome
                        if v < alpha:
                            return (v, bestAction)
                        # Update beta if necessary
                        beta = min(beta, v)

                # We are not at the last ghost in the depth level
                else:
                    # Get list of ghost actions
                    v = float('inf')
                    ghostActions = currState.getLegalActions(currIndex)

                    for action in ghostActions:
                        successor = currState.generateSuccessor(currIndex, action) # generate the successor state
                        v2 = minimax(successor, depth - 1, currAgent + 1, alpha, beta)[0] # compute the minimax value of the next state
                        
                        if v2 < v:
                            bestAction = action
                            v = v2
                
                        # Prune if max can force a better outcome
                        if v < alpha:
                            return (v, bestAction)
                        # Check if the minimax value of the successor is better than the beta
                        beta = min(beta, v)
                return (v, bestAction)
        
        totalDepth = self.depth * gameState.getNumAgents()
        return minimax(gameState, totalDepth, 0, float("-inf"), float("inf"))[1]

        # util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def eminimax(currState, depth, currAgent, alpha):

            # Check terminal state
            if (depth == 0 or currState.isWin() or currState.isLose()):
                return self.evaluationFunction(currState)
    
            # Check the type of the current agent
            currIndex = currAgent

            # If we are at a pacman:
            if currIndex == 0:
                # Get list of pacman actions
                pacmanActions = currState.getLegalActions(0)

                # Loop over actions
                for action in pacmanActions:
                    successor = currState.generateSuccessor(0, action)
                    v2 = eminimax(successor, depth - 1, currAgent + 1, alpha) # Calculate minimax value for the first ghost

                    # Check if the value if the minimax successor val is better than alpha
                    if v2 > alpha:
                        alpha = v2

                return alpha
            
            # If we are at a ghost:
            else:
                totalVal = 0
                # Check if it is the last ghost
                if currIndex == (currState.getNumAgents() - 1):

                    # Get list of ghost actions
                    ghostActions = currState.getLegalActions(currIndex)
                    for action in ghostActions:
                        successor = currState.generateSuccessor(currIndex, action)
                        totalVal += eminimax(successor, depth - 1, 0, alpha) # Calculate the minimax value for the pacman 

                # We are not at the last ghost in the depth level
                else:
                    # Get list of ghost actions
                    ghostActions = currState.getLegalActions(currIndex)
                    for action in ghostActions:
                        successor = currState.generateSuccessor(currIndex, action)
                        totalVal += eminimax(successor, depth - 1, currAgent + 1, alpha) # Calculate the minimax value for the next ghost

                return (totalVal / len(currState.getLegalActions(currIndex)))
            
        totalDepth = self.depth * gameState.getNumAgents()
        bestVal = float("-inf")
        for action in gameState.getLegalActions():
            successor = gameState.generateSuccessor(0, action)
            childVal = eminimax(successor, totalDepth - 1, 1, float("-inf"))
            # Check if eminimax value of the child is the best
            if childVal > bestVal:
                bestVal = childVal
                bestAction = action
        return bestAction
        # util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: This evaulation function uses the following parameters to return a value for each state:
        - numFood: number of foods in the state
        - closestGhost: the Manhattan distance from the Pacman to the closest ghost
        - scaredTimes: a list of the number of moves that each ghost will remain scared
        - closestFood: the Manhattan distance from the Pacman to the closest food
        - len(capsuleCoordinates): the number of capsules in the state
        - currentGameState.getScore(): the built-in .getScore function to evaluate a state

    As a general principle, we want to minimize numFood, maximize closestGhost, minimize closestFood, and minimize len(capsuleCoordinates).
    First, we check if numFood of a given state is 0. If this is the case, we add a large positive reward 
    of 1000000000 to our weighted sum of parameters.
    Next, we check if the closestGhost of a given state is 0. If this is true, we add a large negative reward 
    of -10000000000 to our same weighted sum of parameters. 
    Then, we check if the ghosts are scared using scaredTimes. If so, we minimize the distance to the ghost in an attempt to eat them.
    Finally, if none of these conditions are met, we return the general weighted sum of 
    
    currentGameState.getScore() + 3 * (1 / closestFood) - (1 / closestGhost) - 100 * (len(capsuleCoordinates))

    This weighted sum prioritizes being close to food, far from the ghost, and desires states in which all of the capsules have been consumed.
    """
    "*** YOUR CODE HERE ***"

    # Get relevant data
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList() # remaining food as a list
    ghostStates = currentGameState.getGhostStates()
    ghostPositions = currentGameState.getGhostPositions()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates] # number of moves that each ghost will remain scared
    numFood = currentGameState.getNumFood()
    numGhosts = len(ghostStates)
    capsuleCoordinates = currentGameState.getCapsules()
    # wallCoordinates = currentGameState.getWalls().asList()


    # Calculates the distance to the closest ghost
    closestGhost = float('inf')
    for ghostPos in ghostPositions:
        newDist = manhattanDistance(ghostPos, pos)
        if newDist < closestGhost:
            closestGhost = newDist
    
    # Calculates distance to closest food
    closestFood = float('inf')
    for f in food:
        newDist = manhattanDistance(f, pos)
        if newDist < closestFood:
            closestFood = newDist

    # Large reward for states where the all of the food is eaten
    if numFood == 0:
        return currentGameState.getScore() - (1 / closestGhost) - 100 * (len(capsuleCoordinates)) + 1000000000
        
    # Bad reward for being at the same state as a ghost
    if closestGhost == 0:
        return currentGameState.getScore() + 3 * (1 / closestFood) - 100 * (len(capsuleCoordinates)) - 10000000000

    # If ghosts are scared, high score == close distance to ghost
    if scaredTimes != [0 for i in range(numGhosts)]:
        return 1 / closestGhost
    

    return currentGameState.getScore() + 3 * (1 / closestFood) - (1 / closestGhost) - 100 * (len(capsuleCoordinates))


    util.raiseNotDefined()
    

# Abbreviation
better = betterEvaluationFunction

import random
import numpy as np

def evaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

    """
    return currentGameState.getScore()


def getAction(gameState, d):
    """
      Returns the minimax action from the current gameState using self.evaluationFunction.
      Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves. 

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

    """
    return minimax(gameState, d)

def minimax(gameState, maxDistance):
  currentAgent = 0
  remainingDistance = maxDistance

  legalActions = gameState.getLegalActions(currentAgent)
  possibleValues = [minValue(gameState.generateSuccessor(currentAgent, action), currentAgent, remainingDistance) for action in legalActions]
  maxActionIndex = np.argmax(possibleValues)
  maxAction = legalActions[maxActionIndex]

  return maxAction

def minValue(gameState, currentAgentIndex, remainingDistance):
  currentAgentIndex = nextAgent(gameState, currentAgentIndex)
  
  if isTerminal(gameState, currentAgentIndex, remainingDistance):
    return evaluationFunction(gameState)
  
  value = 99999
  legalActions = gameState.getLegalActions(currentAgentIndex)
  for action in legalActions:
    nextState = gameState.generateSuccessor(currentAgentIndex, action)
    if currentAgentIndex == gameState.getNumAgents() -1:
      #Last ghost, min of max of pacman
      value = min(value, maxValue(nextState, currentAgentIndex, remainingDistance - 1))
    else:
      value = min(value, minValue(nextState, currentAgentIndex, remainingDistance))

  return value

def maxValue(gameState, currentAgentIndex, remainingDistance):
  currentAgentIndex = nextAgent(gameState, currentAgentIndex)
 
  if isTerminal(gameState, currentAgentIndex, remainingDistance):
    return evaluationFunction(gameState)
  
  value = -99999
  legalActions = gameState.getLegalActions(currentAgentIndex)
  for action in legalActions:
    nextState = gameState.generateSuccessor(currentAgentIndex, action)
    value = max(value, minValue(nextState, currentAgentIndex, remainingDistance))

  return value

def nextAgent(gameState, currentAgentIndex):
  return (currentAgentIndex + 1) % gameState.getNumAgents()

def isTerminal(gameState, agentIndex, currentDistance):
  return currentDistance == 0 or gameState.isWin() or gameState.isLose() or len(gameState.getLegalActions(agentIndex)) == 0
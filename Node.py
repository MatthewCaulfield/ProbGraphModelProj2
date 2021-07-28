import numpy
import random
class Node:
    #Node is a class that is made up of the name of the node its states, a CPT, its current state, its parents, its
    #children and its beta values for each state.
    # to initialize a node you need to give it its name and states
    # its parents can be set with setParents(parents, CPTS) where the parent nodes and the CPT given its parents
    # are provided. Its children can be set with setChildren(Children) where the children nodes are given
    def __init__(self, name, states):
        self.states = states
        self.name = name
        self.children = []
        self.parProbability = []
        self.currentState = None
        self.parents = []
        self.beta = [random.uniform(0,1) for i in states]
        self.beta = self.normList(self.beta)

    # setParents set the parents of the node and the CPT of the node given its parents. The order of the parent nodes
    # must match the order they are in in the CPT
    def setParents(self, parents, parentProbs):
        self.parents = parents
        self.parProbability = parentProbs

    # sets the children of the node given a list of children nodes
    def setChildren(self, children):
        self.children = children

    #returns the number of states the node has
    def numStates(self):
        return len(self.states)

    # returns the current state of the parent nodes of this node
    def getParentStates(self):
        if len(self.parents) == 0:
            return []
        elif len(self.parents) == 1:
            return [self.parents[0].currentState]
        else:
            return [self.parents[0].currentState, self.parents[1].currentState]

    # sets the current state of the node
    def setCurrentState(self, state):
        self.currentState = state

    # returns a string with the nodes name and its current state
    def nodeString(self):
        return self.name, self.currentState

    # returns the PDF of the nodes states given its parent's current states
    def getProbGivenParent(self, parentStates):
        if parentStates == [] or parentStates == [None] or parentStates == None:
            probabilities = []
            for i in range(self.numStates()):
                probabilities = probabilities + [self.parProbability[i][0]]
            return probabilities
        else:
            if len(self.parents) == 1:
                probabilities = []
                parentStateIndex = self.parents[0].states.index(parentStates[0])
                for i in range(self.numStates()):
                    parProb = self.parProbability
                    probDistr = parProb[i]
                    probabilities = probabilities + [probDistr[parentStateIndex]]
                return probabilities
            else:
                parIndex = []
                parIndex = parIndex + [self.parents[0].states.index(parentStates[0])]
                parIndex = parIndex + [self.parents[1].states.index(parentStates[1])]
                numStatesPar1 = self.parents[1].numStates()
                probInd = parIndex[1] + parIndex[0]*numStatesPar1
                probabilities = []
                for i in range(self.numStates()):
                    probabilities = probabilities + [self.parProbability[i][probInd]]
                return probabilities

    #sets the beta of the provided state of the node with a given beta
    def setBeta(self, betai, state):
        stateInd = self.states.index(state)
        self.beta[stateInd] = betai

    #calculates the q of the nodes state for mean field
    def qxithetai(self, stateK):
        K = self.numStates()
        stateInd = self.states.index(stateK)
        if K-1 != stateInd:
            q = self.beta[stateInd]
        else:
            totalTheta = 0
            for i in range(K-1):
                totalTheta += self.beta[i]
            q = 1-totalTheta
        return q

    #calculates the total Q of the node for all states
    def getQtotal(self):
        q = 1
        for state in self.states:
            q = q*(self.qxithetai(state))
        return q

    # Finds the probability given the markov blanket for the node with no input.
    def probMarkovBlanket(self):
        parentStates = self.getParentStates()
        pgivenPar = self.getProbGivenParent(parentStates)
        pChild = 1
        for child in self.children:
            childParentStates = child.getParentStates()
            childgivenParentsProb = child.getProbGivenParent(childParentStates)
            childStates = child.states
            childStateInd = childStates.index(child.currentState)
            pChild = pChild*childgivenParentsProb[childStateInd]
        sumPxnPyn = 0
        for prob in pgivenPar:
            sumPxnPyn = sumPxnPyn + prob*pChild
        return [x * pChild / sumPxnPyn for x in pgivenPar]

    # samples the node and sets the current state of the node given its PDF and a random number between 0-1
    def sample(self, randNum, probabilities):
        numStates = self.numStates()
        if numStates == 2:
            if randNum <= probabilities[0]:
                currState = self.states[0]
            else:
                currState = self.states[1]
        elif numStates == 3:
            if randNum <= probabilities[0]:
                currState = self.states[0]
            elif randNum <= (probabilities[0]+probabilities[1]):
                currState = self.states[1]
            else:
                currState = self.states[2]
        elif numStates == 4:
            if randNum <= probabilities[0]:
                currState = self.states[0]
            elif randNum <= (probabilities[0]+probabilities[1]):
                currState = self.states[1]
            elif randNum <= (probabilities[0] + probabilities[1] + probabilities[2]):
                currState = self.states[2]
            else:
                currState = self.states[3]
        elif numStates == 5:
            if randNum <= probabilities[0]:
                currState = self.states[0]
            elif randNum <= (probabilities[0]+probabilities[1]):
                currState = self.states[1]
            elif randNum <= (probabilities[0] + probabilities[1] + probabilities[2]):
                currState = self.states[2]
            elif randNum <= (probabilities[0] + probabilities[1] + probabilities[2] + probabilities[3]):
                currState = self.states[3]
            else:
                currState = self.states[4]
        elif numStates == 6:
            if randNum <= probabilities[0]:
                currState = self.states[0]
            elif randNum <= (probabilities[0]+probabilities[1]):
                currState = self.states[1]
            elif randNum <= (probabilities[0] + probabilities[1] + probabilities[2]):
                currState = self.states[2]
            elif randNum <= (probabilities[0] + probabilities[1] + probabilities[2] + probabilities[3]):
                currState = self.states[3]
            elif randNum <= (probabilities[0] + probabilities[1] + probabilities[2] + probabilities[3] + probabilities[4]):
                currState = self.states[4]
            else:
                currState = self.states[5]
        self.currentState = currState
        return currState

    # implementation of the map function when given a pdf as an input
    def map(self, probabilites):
        index = probabilites.index(max(probabilites))
        return self.states[index]

    # normalizes the values of a list
    def normList(self, myList):
        alpha = 0
        for num in myList:
            alpha += num
        return [x / alpha for x in myList]
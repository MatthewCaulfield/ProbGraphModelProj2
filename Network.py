import random
import numpy

class Network:

    # Network class takes a set of nodes in topographical order  as an input and creates a network out of them


    def __init__(self, Nodes):
        self.Nodes = Nodes
        self.numNodes = len(self.Nodes)

    # Implementation of the likelihood weighted sampling approximate inference
    # input given which is a tuple of lists like ([DuctFlow, CardiacMixing], ['None', 'Mild'])
    # The first list is a list of nodes the second list is the given state of each node in
    # the same order as the list of nodes.
    # Other input is numSamples which is the number of samples to take
    # returns the samples and their weights in form of list [(sample0 ,weight0), ..., (sampleN, weightN)]
    def likelihoodWeightedSampling(self, given, numSamples):
        weights = []
        samples = []
        for i in range(numSamples):
            weights = weights + [1]

        for t in range(numSamples):
            currSample = []
            for n in range(self.numNodes):
                rand = random.uniform(0, 1)
                currNode = self.Nodes[n]
                piXn = currNode.getParentStates()
                PNodeGivenParents = currNode.getProbGivenParent(piXn)
                if currNode in given[0]:
                    nodeInd = given[0].index(currNode)
                    currNode.setCurrentState(given[1][nodeInd])
                    currSample = currSample+[(currNode.name, given[1][nodeInd])]
                    currSampleStateInd = currNode.states.index(given[1][nodeInd])
                    weights[t] = weights[t]*PNodeGivenParents[currSampleStateInd]
                else:
                    currSample = currSample + [(currNode.name, currNode.sample(rand, PNodeGivenParents))]
            samples = samples + [(currSample, weights[t])]
        return samples

    # Implementation of gibbs sampling
    # inputs are given which is a tuple of lists like ([DuctFlow, CardiacMixing], ['None', 'Mild'])
    # The first list is a list of nodes the second list is the given state of each node in
    # the same order as the list of nodes.
    # burn in length which is the number of burn in samples
    # numSamples which is the number of recorded samples before skipping samples
    # initialStates which are the initial states of the nodes in the same topographical order
    # of the network
    # skipTime is how many samples to skip between each recorded sample
    def gibbsSampling(self, given, burnInLength, numSamples, initialStates, skipTime):
        numNodes = len(self.Nodes)
        currsample = []
        sample = []
        i = 0
        for state in initialStates:
            state[0].setCurrentState(state[1])
            currsample = currsample + [(state[0].name, state[1])]
            i += 1

        for i in range(burnInLength):
            randUniform = random.uniform(0, 1)
            n = random.randint(0, numNodes-1)
            currNode = self.Nodes[n]
            if currNode in given[0]:
                nodeInd = given[0].index(currNode)
                currNode.currentState = given[1][nodeInd]
                currsample[n] = (currNode.name, given[1][nodeInd])
            else:
                markBlanket = currNode.probMarkovBlanket()
                currsample[n] = (currNode.name, currNode.sample(randUniform, markBlanket))

        for i in range(numSamples):
            randUniform = random.uniform(0,1)
            n = random.randint(0, numNodes-1)
            currNode = self.Nodes[n]
            if currNode in given[0]:
                nodeInd = given[0].index(currNode)
                currNode.currentState = given[1][nodeInd]
                currsample[n] = (currNode.name, given[1][nodeInd])
            else:
                markBlanket = currNode.probMarkovBlanket()
                currsample[n] = (currNode.name, currNode.sample(randUniform, markBlanket))
            sample.append(currsample.copy())

        sampleSkip = sample[::skipTime]
        return sampleSkip

    # mean field returns the beta for the network also sets the beta_N for each node
    # input is given given which is a tuple of lists like ([DuctFlow, CardiacMixing], ['None', 'Mild'])
    # The first list is a list of nodes the second list is the given state of each node in
    # the same order as the list of nodes.
    def meanField(self, given):
        betaOld = []
        beta = []
        for node in self.Nodes:
            for state in node.states:
                betaOld.append(1)
                nodeStateInd = node.states.index(state)
                beta.append(node.beta[nodeStateInd])

        converge = self.convergence(betaOld, beta)
        while(converge > 10^(-15)):
            i = 0
            betaOld = beta

            for node in self.Nodes:
                if node not in given[0]:
                    fracBottom = 0
                    for state in node.states:
                            fracBottom = fracBottom + numpy.exp(self.expectation(node, state))
                    for state in node.states:
                        fracTop = numpy.exp(self.expectation(node, state))
                        betaI = fracTop/fracBottom
                        beta[i] = betaI
                        node.setBeta(betaI, state)
                        i += 1
                else:
                    for state in node.states:
                        givenIndex = given[0].index(node)
                        givenState = given[1][givenIndex]
                        if state == givenState:
                            betaI = 1
                            beta[i] = betaI
                            node.setBeta(betaI, state)
                            i += 1
                        else:
                            betaI = 0
                            beta[i] = betaI
                            node.setBeta(betaI, state)
                            i += 1
            converge = self.convergence(betaOld, beta)
        return beta

    # returns the expectation of NodeI and its StateI over q(X/xn) like in formula 3.50 and 3.49 in the book
    def expectation(self, NodeI, stateI):
        nodes = self.Nodes.copy()
        nodePar = NodeI.parents.copy()
        numParents = len(nodePar)
        expected = 0
        if numParents == 0:
            for node in nodes:
                if node != NodeI:
                    q = node.getQtotal()
                    prob = NodeI.getProbGivenParent(NodeI.getParentStates())
                    probInd = NodeI.states.index(stateI)
                    expected = expected + q*numpy.log(prob[probInd])
        elif numParents == 1:
            nodes.remove(nodePar[0])
            for i in range(len(nodePar[0].states)):
                for node in nodes:
                    if node != NodeI:
                            q = node.getQtotal()
                            prob = NodeI.getProbGivenParent([nodePar[0].states[i]])
                            probInd = NodeI.states.index(stateI)
                            expected = expected + q * numpy.log(prob[probInd])
        elif numParents == 2:
            nodes.remove(nodePar[0])
            nodes.remove(nodePar[1])
            for i in range(len(nodePar[0].states)):
                for j in range(len(nodePar[1].states)):
                    for node in nodes:
                        if node != NodeI:
                            q = node.getQtotal()
                            prob = NodeI.getProbGivenParent([nodePar[0].states[i], nodePar[1].states[j]])
                            probInd = NodeI.states.index(stateI)
                            expected = expected + q * numpy.log(prob[probInd])
        return expected

    # checks if two lists are converging
    def convergence(self, listOne, listTwo):
        if len(listOne) != len(listTwo):
            print("Lists different lengths")
            return None
        else:
            difference = 0
            for i in range(len(listOne)):
                difference += (listOne[i]-listTwo[i])
            return difference

    #normalizes a list
    def normList(self, myList):
        alpha = 0
        for num in myList:
            alpha += num
        return [x / alpha for x in myList]
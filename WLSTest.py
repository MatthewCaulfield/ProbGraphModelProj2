import numpy
import Node as nd
import Network as ntwrk
import pickle
#import probMethods as pm
with open('parameter.pkl', 'rb') as fp:
    CPTs = pickle.load(fp)

BirthAsphyxia = nd.Node('BirthAsphyxia', ['yes', 'no'])
BirthAsphyxia.setParents([], CPTs['BirthAsphyxia'])
Disease = nd.Node('Disease', ['PFC', 'TGA', 'Fallot', 'PAIVS', 'TAPVD”', 'Lung'])
Disease.setParents([BirthAsphyxia], CPTs['Disease'])
DuctFlow = nd.Node('DuctFlow', ['Lt to Rt', 'None', 'Rt to Lt'])
DuctFlow.setParents([Disease], CPTs['DuctFlow'])
CardiacMixing = nd.Node('CardiacMixing', ['None', 'Mild', 'Complete', 'Transparent'])
CardiacMixing.setParents([Disease], CPTs['CardiacMixing'])
HypDistrib = nd.Node('HypDistrib', ['equal', 'unequal'])
HypDistrib.setParents([DuctFlow, CardiacMixing], CPTs['HypDistrib'])


childNet = ntwrk.Network([BirthAsphyxia, Disease, DuctFlow, CardiacMixing, HypDistrib])
testSamples = childNet.likelihoodWeightedSampling(([HypDistrib], ['equal']), 100)

print(testSamples)

PBirthAphyxWLS = [0, 0]
PDiseaseWLS = [0, 0, 0, 0, 0, 0]
for i in range(100):
    if ('BirthAsphyxia', 'yes') in testSamples[i][0]:
        PBirthAphyxWLS[0] += testSamples[i][1]
    elif('BirthAsphyxia', 'no') in testSamples[i][0]:
        PBirthAphyxWLS[1] += testSamples[i][1]
    if ('Disease', 'PFC') in testSamples[i][0]:
        PDiseaseWLS[0]+= testSamples[i][1]
    elif ('Disease', 'TGA') in testSamples[i][0]:
        PDiseaseWLS[1]+= testSamples[i][1]
    elif ('Disease', 'Fallot') in testSamples[i][0]:
        PDiseaseWLS[2]+= testSamples[i][1]
    elif ('Disease', 'PAIVS') in testSamples[i][0]:
        PDiseaseWLS[3]+= testSamples[i][1]
    elif ('Disease', 'TAPVD') in testSamples[i][0]:
        PDiseaseWLS[4]+= testSamples[i][1]
    elif ('Disease', 'Lung') in testSamples[i][0]:
        PDiseaseWLS[5] += testSamples[i][1]

PDiseaseWLS = childNet.normList(PDiseaseWLS)
PBirthAphyxWLS = childNet.normList(PBirthAphyxWLS)
print(PBirthAphyxWLS)
print('p(Birth Asphyxia=yes | HypDistr = unequel) = ', PBirthAphyxWLS[0])
print(PDiseaseWLS)
print('Disease =', Disease.map(PDiseaseWLS))

#print(BirthAsphyxia.getProbGivenParent([]))
#disProb = Disease.getProbGivenParent(['yes'])
#ducProb = DuctFlow.getProbGivenParent(['TGA'])
#cardProb = CardiacMixing.getProbGivenParent(['TGA'])
#DuctFlow.currentState = 'Lt to Rt'
#CardiacMixing.currentState = 'Mild'
#hypParStates = HypDistrib.getParentStates()
#hypProb = HypDistrib.getProbGivenParent(hypParStates)
#print(hypProb)
#print(ducProb)
#print(cardProb)

#prob = CPTs['LowerBodyO2'][0]

#print(CPTs['BirthAsphyxia'])
#print(CPTs['DuctFlow'])
#print(CPTs['CardiacMixing'])
#print(CPTs['HypDistrib'])
#print(prob)
print(CPTs['Disease'])
#print(CPTs)
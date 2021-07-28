import numpy
import Node as nd
import Network as ntwrk
import pickle
#import probMethods as pm
with open('parameter.pkl', 'rb') as fp:
    CPTs = pickle.load(fp)

BirthAsphyxia = nd.Node('BirthAsphyxia', ['yes', 'no'])
BirthAsphyxia.setParents([], CPTs['BirthAsphyxia'])
Disease = nd.Node('Disease', ['PFC', 'TGA', 'Fallot', 'PAIVS', 'TAPVD', 'Lung'])
Disease.setParents([BirthAsphyxia], CPTs['Disease'])
LVH = nd.Node('LVH', ['yes', 'no'])
LVH.setParents([Disease], CPTs['LVH'])
DuctFlow = nd.Node('DuctFlow', ['Lt to Rt', 'None', 'Rt to Lt'])
DuctFlow.setParents([Disease], CPTs['DuctFlow'])
CardiacMixing = nd.Node('CardiacMixing', ['None', 'Mild', 'Complete', 'Transparent'])
CardiacMixing.setParents([Disease], CPTs['CardiacMixing'])
LungParenchema = nd.Node('LungParench', ['Normal', 'Oedema', 'Abnormal'])
LungParenchema.setParents([Disease], CPTs['LungParench'])
LungFlow = nd.Node('LungFlow', ['Normal', 'Low', 'High'])
LungFlow.setParents([Disease], CPTs['LungFlow'])
Sick = nd.Node('Sick', ['yes', 'no'])
Sick.setParents([Disease], CPTs['Sick'])
Age = nd.Node('Age', ['0-3 days', '4-10 days', '11-30 days'])
Age.setParents([Disease, Sick], CPTs['Age'])
HypDistrib = nd.Node('HypDistrib', ['equal', 'unequal'])
HypDistrib.setParents([DuctFlow, CardiacMixing], CPTs['HypDistrib'])
HypoxiaInO2 = nd.Node('HypoxiaInO2', ['None', 'Moderate', 'Severe'])
HypoxiaInO2.setParents([CardiacMixing, LungParenchema], CPTs['HypoxiaInO2'])
CO2 = nd.Node('CO2', ['Normal', 'Low', 'High'])
CO2.setParents([LungParenchema], CPTs['CO2'])
ChestXray = nd.Node('ChestXray', ['Normal', 'Oligaemic', 'Plethoric', 'Grd._Glass', 'Asy/Patchy'])
ChestXray.setParents([LungParenchema, LungFlow], CPTs['ChestXray'])
Grunting = nd.Node('Grunting', ['yes', 'no'])
Grunting.setParents([LungParenchema, Sick], CPTs['Grunting'])
LVHreport = nd.Node('LVHreport', ['yes', 'no'])
LVHreport.setParents([LVH], CPTs['LVHreport'])
LowerBodyO2 = nd.Node('LowerBodyO2', ['<5', '5-12', '12+'])
LowerBodyO2.setParents([HypDistrib, HypoxiaInO2], CPTs['LowerBodyO2'])
RUQO2 = nd.Node('RUQO2', ['<5', '5-12', '12+'])
RUQO2.setParents([HypoxiaInO2], CPTs['RUQO2'])
CO2Report = nd.Node('CO2Report', ['<7.5', '>=7.5'])
CO2Report.setParents([CO2], CPTs['CO2Report'])
XrayReport = nd.Node('XrayReport', ['Normal', 'Oligaemic', 'Plethoric', 'Grd._Glass', 'Asy/Patchy'])
XrayReport.setParents([ChestXray], CPTs['XrayReport'])
GruntingReport = nd.Node('GruntingReport', ['yes', 'no'])
GruntingReport.setParents([Grunting], CPTs['GruntingReport'])

BirthAsphyxia.setChildren([Disease])
Disease.setChildren([LVH, DuctFlow, CardiacMixing, LungParenchema, LungFlow, Sick, Age])
LVH.setChildren([LVHreport])
DuctFlow.setChildren([HypDistrib])
CardiacMixing.setChildren([HypDistrib, HypoxiaInO2])
LungParenchema.setChildren([HypoxiaInO2, CO2, ChestXray, Grunting])
LungFlow.setChildren([ChestXray])
Sick.setChildren([Age, Grunting])
HypDistrib.setChildren([LowerBodyO2])
HypoxiaInO2.setChildren([LowerBodyO2, RUQO2])
CO2.setChildren([CO2Report])
ChestXray.setChildren([XrayReport])
Grunting.setChildren([GruntingReport])

orderedNodeList= [BirthAsphyxia, Disease, LVH, DuctFlow, CardiacMixing, LungParenchema, LungFlow, Sick, Age, HypDistrib,
                   HypoxiaInO2, CO2, ChestXray, Grunting, LVHreport, LowerBodyO2, RUQO2, CO2Report, XrayReport,
                  GruntingReport]

childNet = ntwrk.Network(orderedNodeList)

given = ([CO2Report, LVHreport, XrayReport], ['<7.5', 'yes', 'Plethoric'])

initialStates = [(BirthAsphyxia, 'yes'), (Disease, 'PFC'), (LVH, 'yes'), (DuctFlow, 'None'), (CardiacMixing, 'None'),
                 (LungParenchema, 'Normal'), (LungFlow,'Low'), (Sick, 'yes'), (Age,'4-10 days'), (HypDistrib, 'equal'),
                 (HypoxiaInO2, 'None'), (CO2, 'Normal'), (ChestXray, 'Plethoric'), (Grunting, 'yes'), (LVHreport,'yes'),
                 (LowerBodyO2, '<5'), (RUQO2, "<5"), (CO2Report, '<7.5'),
                 (XrayReport, 'Plethoric'), (GruntingReport, 'yes')]


burnInLength = 1000
numSamplesGib = 10000
skipTime = 10

testSamples = childNet.gibbsSampling(given, burnInLength, numSamplesGib, initialStates, skipTime)




#print('\n'.join(map(str, testSamples)))
PBirthAphyxGS = [0, 0]
PDiseaseGS = [0, 0, 0, 0, 0, 0]
for i in range(round(numSamplesGib/skipTime)):
    if ('BirthAsphyxia', 'yes') in testSamples[i]:
        PBirthAphyxGS[0] += 1
    elif('BirthAsphyxia', 'no') in testSamples[i]:
        PBirthAphyxGS[1] += 1
    if ('Disease', 'PFC') in testSamples[i]:
        PDiseaseGS[0]+= 1
    elif ('Disease', 'TGA') in testSamples[i]:
        PDiseaseGS[1]+= 1
    elif ('Disease', 'Fallot') in testSamples[i]:
        PDiseaseGS[2]+= 1
    elif ('Disease', 'PAIVS') in testSamples[i]:
        PDiseaseGS[3]+= 1
    elif ('Disease', 'TAPVD') in testSamples[i]:
        PDiseaseGS[4]+= 1
    elif ('Disease', 'Lung') in testSamples[i]:
        PDiseaseGS[5] += 1
print("Gibbs%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
PDiseaseGS = childNet.normList(PDiseaseGS)
PBirthAphyxGS = childNet.normList(PBirthAphyxGS)
#print(PBirthAphyxGS)
print('p(Birth Asphyxia=yes |CO2report =<7.5, LVHReport=yes, and X-rayReport=Plethoric) = ', PBirthAphyxGS[0])
#print(PDiseaseGS)
print('Disease =', Disease.map(PDiseaseGS))



numSamples = 10000
testSamplesWLS = childNet.likelihoodWeightedSampling(given, numSamples)
weight = 0
PBirthAphyxWLS = [0, 0]
PDiseaseWLS = [0, 0, 0, 0, 0, 0]
for i in range(numSamples):
    weight += testSamplesWLS[i][1]
    if ('BirthAsphyxia', 'yes') in testSamplesWLS[i][0]:
        PBirthAphyxWLS[0] += testSamplesWLS[i][1]
    elif('BirthAsphyxia', 'no') in testSamplesWLS[i][0]:
        PBirthAphyxWLS[1] += testSamplesWLS[i][1]
    if ('Disease', 'PFC') in testSamplesWLS[i][0]:
        PDiseaseWLS[0]+= testSamplesWLS[i][1]
    elif ('Disease', 'TGA') in testSamplesWLS[i][0]:
        PDiseaseWLS[1]+= testSamplesWLS[i][1]
    elif ('Disease', 'Fallot') in testSamplesWLS[i][0]:
        PDiseaseWLS[2]+= testSamplesWLS[i][1]
    elif ('Disease', 'PAIVS') in testSamplesWLS[i][0]:
        PDiseaseWLS[3]+= testSamplesWLS[i][1]
    elif ('Disease', 'TAPVD') in testSamplesWLS[i][0]:
        PDiseaseWLS[4]+= testSamplesWLS[i][1]
    elif ('Disease', 'Lung') in testSamplesWLS[i][0]:
        PDiseaseWLS[5] += testSamplesWLS[i][1]


#PDiseaseWLS = childNet.normList(PDiseaseWLS)
#PBirthAphyxWLS = childNet.normList(PBirthAphyxWLS)
print("Weighted%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
#print(PBirthAphyxWLS)
print('p(Birth Asphyxia=yes | CO2report =<7.5, LVHReport=yes, and X-rayReport=Plethoric) = ', PBirthAphyxWLS[0]/weight)
#print(PDiseaseWLS)
print('Disease =', Disease.map(PDiseaseWLS))

print("Mean Field%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

testSamples = childNet.meanField(given)
#print(BirthAsphyxia.beta)
print('p(Birth Asphyxia=yes | CO2report =<7.5, LVHReport=yes, and X-rayReport=Plethoric) = ', BirthAsphyxia.qxithetai('yes'))
#print(Disease.beta)
print('Disease =', Disease.map(Disease.beta))



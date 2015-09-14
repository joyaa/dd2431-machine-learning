"""
DD2431 HT15
Lab 1
"""
import monkdata as m
import dtree as tree
import drawtree as draw
import random

"3 ENTROPY"
"--Assignment 1"
print("--------------------------------------")
print("Assignment 1: Entropy of training sets")
monkEntropy = [round(tree.entropy(m.monk1)), round(tree.entropy(m.monk2)), round(tree.entropy(m.monk3))]
"--Answer to Assignment 1"
print(monkEntropy, "\n") 

"4 INFORMATION GAIN"
"--Assignment 2"
monkTrainingSets = [m.monk1, m.monk2, m.monk3]
informationGain = [];

print("Assignment 2: Expected information gains")
att = []; #save values for each attribute
for monk in monkTrainingSets:	#for each data set
	for attribute in m.attributes:	#for every attribute
		#calculate the gain of splitting by the attribute
		att.append(round(tree.averageGain(monk, attribute),5))

	informationGain.append(att) #save a "row vector" 
	att=[]

"--Answer to Assignment 2"
print(informationGain[2], "\n")

#print(tree.bestAttribute(m.monk1, m.attributes))

""" 
Attribute a5 has the largest information gain meaning that it reduces the 
uncertainty the most. Thus, it should be used for splitting at the root node.
"""


"5 BUILDING DECISION TREES"
sel=[]
for i in range(4): #splits data into subset according to attr a5
	sel.append(tree.select(m.monk1, m.attributes[4], m.attributes[4].values[i]))

#print(sel)
sub = []
mC = []
for subset in sel:
	for i in [0,1,2,3,5]:
		sub.append(tree.averageGain(subset, m.attributes[i]));
	mC.append(tree.mostCommon(subset));
	
	#print(sub)	
	sub = []

"Highest information gain on second level of the tree # 2 - A4 , 3 - A6 , 4 - A1 #"

"""Assignment 3"""
tree1 = tree.buildTree(m.monk1, m.attributes)
tree2 = tree.buildTree(m.monk2, m.attributes)
tree3 = tree.buildTree(m.monk3, m.attributes)

#draw.drawTree(tree1)
#draw.drawTree(tree2)
#draw.drawTree(tree3)

print("Assignment 3: Decision tree performances")

print("Train errors:")
print(round(tree.check(tree1, m.monk1),5))
print(round(tree.check(tree2, m.monk2),5))
print(round(tree.check(tree3, m.monk3),5))

print("Test errors:")
print(round(tree.check(tree1, m.monk1test),5))
print(round(tree.check(tree2, m.monk2test),5))
print(round(tree.check(tree3, m.monk3test),5))

#print(tree.mostCommon(tree.select(sel[3],m.attributes[0],3)));

"6 PRUNING"
def partition(data, fraction):
	ldata = list(data)
	#random.shuffle(ldata)
	breakPoint = int(len(ldata) * fraction)
	return ldata[:breakPoint], ldata[breakPoint:]

print("Assignment 4: ")

"--Assignment 4"
fractionError = []
fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
for monk in [m.monk1, m.monk3]:
	tempErrors = []
	for f in fractions:
		m1train, m1val = partition(monk, f)

		t1 = tree.buildTree(m1train, m.attributes)
		currentPerf = tree.check(t1, m1val) 
		perf1 = 0
		while perf1 < currentPerf:
			perf1 = currentPerf
			for pTree in tree.allPruned(t1):
				temp = tree.check(pTree, m1val)
				if currentPerf < temp:
					currentPerf = temp
					t1 = pTree
		#print([perf1,f])
		tempErrors.append(round(perf1, 5))
	fractionError.append(tempErrors)

print(fractionError)
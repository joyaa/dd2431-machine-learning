"""
DD2431 HT15
Lab 1
"""
import monkdata as m
import dtree as tree
import drawtree as draw
import random

#Assignment 1
#print ("Assignment 1: Entropy of training sets")
monkEntropy = [tree.entropy(m.monk1), tree.entropy(m.monk2), tree.entropy(m.monk3)]
#print (monkEntropy)


monkTrainingSets = [m.monk1, m.monk2, m.monk3] #kan även användas i assignment 1
#Assignment 2
#print("Assignment 2: Expected information gains")

#informationGain = [[0 for x in range(monkTrainingSets)] for x in range(m.attributes)] 
informationGain = [];
att= [];
i = 0; j = 0;
for monk in monkTrainingSets:
	for attribute in m.attributes:
		
		att.append(tree.averageGain(m.monk1, attribute))

	informationGain.append(att)
	#print(att)
	att=[]
#print(informationGain[0])
#print(tree.bestAttribute(m.monk1,m.attributes))

""" 
Attribute a5 has the largest information gain meaning that it reduces the 
uncertainty the most. Thus, it should be used for splitting at the root node.
"""


#5 BUILDING DECISION TREES
#for monk1

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
print(mC)
# Highest information gain on second level of the tree # 2 - A4 , 3 - A6 , 4 - A1 #


"""Assignment 3"""
tree1 = tree.buildTree(m.monk1, m.attributes)
tree2 = tree.buildTree(m.monk2, m.attributes)
tree3 = tree.buildTree(m.monk3, m.attributes)
"""
print("Train errors:")
print(tree.check(tree1, m.monk1))
print(tree.check(tree2, m.monk2))
print(tree.check(tree3, m.monk3))

print("Test errors:")
print(tree.check(tree1, m.monk1test))
print(tree.check(tree2, m.monk2test))
print(tree.check(tree3, m.monk3test))
"""
#draw.drawTree(tree1)

#print(tree.mostCommon(tree.select(sel[3],m.attributes[0],3)));

#6 PRUNING

def partition(data, fraction):
	ldata = list(data)
	#random.shuffle(ldata)
	breakPoint = int(len(ldata) * fraction)
	return ldata[:breakPoint], ldata[breakPoint:]

monk1train, monk1val = partition(m.monk1, 0.6)

#tree1

t1 = tree.buildTree(monk1train, m.attributes)
currentPerf = tree.check(t1, monk1val) 
perf1 = 0
while perf1 < currentPerf:
	perf1 = currentPerf
	for pTree in tree.allPruned(t1):
		temp = tree.check(pTree, monk1val)
		if currentPerf < temp:
			currentPerf = temp
			t1 = pTree
	#print(perf1)



	
#Assignment 4
fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

for f in fractions:
	m1train, m1val = partition(m.monk1, f)

	t1 = tree.buildTree(m1train, m.attributes)
	currentPerf = tree.check(t1, m1val) 
	perf1 = 0
	while perf1 < currentPerf:
		perf1 = currentPerf
		for pTree in tree.allPruned(t1):
			temp = tree.check(pTree, monk1val)
			if currentPerf < temp:
				currentPerf = temp
				t1 = pTree
	print([perf1,f])


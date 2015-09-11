"""
DD2431 HT15
Lab 1
"""
import monkdata as m
import dtree as tree

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

"""Assignment 3"""

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
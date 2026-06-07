#class for the nodes
class treeNode():
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.parent = parentNode     
        self.children = {}
        
    def inc(self, numOccur):
        self.count += numOccur

    def dec(self, numOccur):
        self.count -= numOccur
            
    def disp(self, ind=1):
        print ('  '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind+1)  

#class for the FP Tree
class DominantTree():
    """
    A frequent pattern tree.
    """
    def __init__(self, transactions, threshold, root_value, root_count):
        """
        Initialize the tree.
        """
        self.frequent = self.find_frequent_items(transactions, threshold)
        self.itemTable = {}
        self.headers = {}
        self.buffer = []
        self.maxBufferLength = 100
        self.maxResursionCall = 50
        self.root = self.createTree(transactions, root_value, root_count, self.frequent, self.headers)

    @staticmethod
    def find_frequent_items(transactions, threshold):
        """
        Create a dictionary of items with occurrences above the threshold.
        """
        items = {}

        for transaction in transactions:
            for item in transaction:
                if item in items:
                    items[item] += 1
                else:
                    items[item] = 1

        #print (items)
        for key in list(items.keys()):
            if items[key] < threshold:
                del items[key]
        
        #print (items)
        return items
    
    def createInitSet(self,dataSet):
        retDict = {}
        for trans in dataSet:
            if frozenset(trans) in retDict:
                retDict[frozenset(trans)] += 1
            else:
                retDict[frozenset(trans)] = 1
            
        return retDict

    def createTree(self, dataSet, root_value, root_count, frequent, linkTable): 
       
        retTree = treeNode(root_value, root_count, None) 
        
        new_records = []
        
        for transaction in dataSet:
            sorted_items = [x for x in transaction if x in frequent]

            sorted_items.sort(key=lambda x: frequent[x], reverse=True)

            new_records.append(sorted_items)
            
        dataSet = self.createInitSet(new_records) #initSet
        
        for tranSet, count in dataSet.items():  
       
            iac = list(self.updateTable (tranSet, count).items()) 
            siac = sorted(iac, key = lambda x: x[1], reverse = True)
        
            
            nodeList = []
            newNodeList = []
            self.updateTree (siac, nodeList, newNodeList, count, retTree, False)   
      
            
            if len(nodeList) != len(iac):
                
                self.buffer.append([tranSet, count])
            else:
                self.increse_support_of_nodes (nodeList, count)
                self.update_header_table (newNodeList)
                
  
        
            if len(self.buffer) > self.maxBufferLength:
            
                self.bufferHandler(retTree)
                
        if len(self.buffer) > 0:
            self.bufferHandler(retTree)
        
        #retTree.tree_pruning (self.threshold)
       
        return retTree
    
    def getFromTable (self, items):
        itemsAndCounts = {}
        for item in items:
            cnt = self.itemTable.get(item)
            itemsAndCounts.update({item: cnt})
        return itemsAndCounts
    
    def bufferHandler (self, retTree):
        while len(self.buffer) > 0:
            it = self.buffer.pop(0)
            tranSet = it[0]
            count = it[1]
                        
            iac = list(self.getFromTable (tranSet).items()) 
            siac = sorted(iac, key = lambda x: x[1], reverse = True)

            nodeList = []
            newNodeList = []
            self.updateTree (siac, nodeList, newNodeList, count, retTree, True)  
            
            self.increse_support_of_nodes (nodeList, count)
            self.update_header_table (newNodeList)
            
    def updateTable (self, items, count):
        itemsAndCounts = {}
        for item in items:
            cnt = self.itemTable.get (item)
            if cnt == None:
                self.itemTable.update({item: count})
                itemsAndCounts.update({item: count})
            else:
                cnt += count
                self.itemTable.update({item: cnt})
                itemsAndCounts.update({item: cnt})
        return itemsAndCounts 
    
    def findDominantNode (self, citems, nl, nnl, inTree, bh): #tu -> table updated

        cntList = []
        
        for item, count in citems:
            cntList.append(count)
            if item in inTree.children:
                nl.append(inTree.children[item])
                citems.remove((item, count))
                return inTree.children[item]
            
        if bh == False:
            NRV = True # Non-repeated-Value
            for i in range (0, len(cntList) - 1):
                if cntList[i] == cntList[i+1]:
                    NRV = False

            if NRV == False:
                return None 
            else:
                item = citems[0][0]
                inTree.children[item] = treeNode(item, 0, inTree)
                nl.append(inTree.children[item])
                nnl.append(inTree.children[item])
                del citems[0]
                return inTree.children[item]
        else:
            item = citems[0][0]
            inTree.children[item] = treeNode(item, 0, inTree)
            nl.append(inTree.children[item])
            nnl.append(inTree.children[item])
            del citems[0]
            return inTree.children[item]

    def updateTree(self, candidateItems, nodeList, newNodeList, count, inTree, bh): 


        if candidateItems is None:
            return inTree
        elif len(candidateItems) == 0:
            return None
        elif len(candidateItems) > self.maxResursionCall:
            candidateItems.clear()
            return None

        DN = self.findDominantNode(candidateItems, nodeList, newNodeList, inTree, bh)
            
        if DN is None:
            return inTree
        else:
            inTree = DN
       
            self.updateTree(candidateItems, nodeList, newNodeList, count, inTree, bh)
        
    def increse_support_of_nodes (self, nodeList, count):
 
        if len(nodeList) < 1:
            return
        
        for node in nodeList:
           
            node.count += count
    
    def update_header_table (self, nodes):
        # add in header structure
        for node in nodes:
            if node.name in self.headers.keys():
                self.headers[node.name].append(node)
            else:
                self.headers[node.name] = [node]
    
    #pattern mining begins...            
    def mine_patterns(self, threshold):
        """
        Mine the constructed FP tree for frequent patterns.
        """
        if self.tree_has_single_path(self.root):
            #print ("True")
            return self.generate_pattern_list()
        else:
            #print ("+True")
            return self.zip_patterns(self.mine_sub_trees(threshold))
        
    def tree_has_single_path(self, node):
        """
        If there is a single path in the tree,
        return True, else return False.
        """
        num_children = len(node.children)
        if num_children > 1:
            return False
        elif num_children == 0:
            return True
        else:
            return True and self.tree_has_single_path(list(node.children.values())[0])
        
    def generate_pattern_list(self):
        """
        Generate a list of patterns with support counts.
        """
        patterns = {}
        items = self.frequent.keys()

        # If we are in a conditional tree,
        # the suffix is a pattern on its own.
        if self.root.name is None:
            suffix_value = []
        else:
            suffix_value = [self.root.name]
            patterns[tuple(suffix_value)] = self.root.count

        for i in range(1, len(items) + 1):
            for subset in itertools.combinations(items, i):
                pattern = tuple(sorted(list(subset) + suffix_value))
                patterns[pattern] = \
                    min([self.frequent[x] for x in subset])

        return patterns
    
    def zip_patterns(self, patterns):
       
        suffix = self.root.name

        if suffix is not None:
            # We are in a conditional tree.
            new_patterns = {}
            for key in patterns.keys():
                new_patterns[tuple(sorted(list(key) + [suffix]))] = patterns[key]

            return new_patterns

        return patterns
    
    def mine_sub_trees(self, threshold):
        
        patterns = {}
        mining_order = sorted(self.frequent.keys(),
                              key=lambda x: self.frequent[x])
        
        for item in mining_order:
            suffixes = []
            conditional_tree_input = []
            
            suffixes = self.headers[item]
            
            for suffix in suffixes:
                frequency = suffix.count
                path = []
                parent = suffix.parent

                while parent.parent is not None:
                    path.append(parent.name)
                    parent = parent.parent
                
                for i in range(frequency):
                    conditional_tree_input.append(path)

            subtree = DominantTree(conditional_tree_input, threshold,
                             item, self.frequent[item])
            #subtree.root.disp()
            subtree_patterns = subtree.mine_patterns(threshold)

            # Insert subtree patterns into main patterns dictionary.
            for pattern in subtree_patterns.keys():
                if pattern in patterns:
                    patterns[pattern] += subtree_patterns[pattern]
                else:
                    patterns[pattern] = subtree_patterns[pattern]

        return patterns


def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

#finding the frequent patterns
def find_frequent_patterns(transactions, support_threshold):
    '''
    Using a set a trasnactions to find patterns in it over 
    the specified support threshold.
    '''
    tree = DominantTree(transactions, support_threshold, None, None)
    pattern = tree.mine_patterns(support_threshold)
    #print("Frequent Patterns: ", pattern)
    return pattern

def generate_association_rules(patterns, confidence_threshold):

    rules = {}
    for itemset in patterns.keys():
        upper_support = patterns[itemset]

        for i in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, i):
                antecedent = tuple(sorted(antecedent))
                consequent = tuple(sorted(set(itemset) - set(antecedent)))

                if antecedent in patterns:
                    lower_support = patterns[antecedent]
                    confidence = float(upper_support) / lower_support

                    if confidence >= confidence_threshold:
                        rules[antecedent] = (consequent, confidence)

    return rules

import itertools
import time
import os
import psutil
import csv
import gc

gc.collect()

# Open file in append mode once, and keep it open throughout the loop
with open('result.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Dataset: zoo.dat'])
    writer.writerow(['Algorithm: D-Tree'])
    
    writer.writerow(['Support threshold', 'Min_supp', 'Time', 'Memory', 'Rule'])

    init_mem = get_process_memory()
    init_time = time.perf_counter()
    
    with open('zoo.dat', 'r') as f:
        records = [line.strip().split() for line in f]
    
    for i in range(100, 39, -10):   
        min_supp = len(records) * (i / 100)
        
        print(str(i) + ", Minimum Support: " + str(min_supp))
        
        patterns = find_frequent_patterns(records, min_supp)
        rules = generate_association_rules(patterns, 0.5)
        
        print("Patterns length: " + str(len(patterns)) + "   Rule length: " + str(len(rules)))

        fin_mem = get_process_memory()
        fin_time = time.perf_counter()

        exec_time = fin_time - init_time
        mem_used = fin_mem - init_mem

        print("Execution Time (Seconds): ", exec_time)
        print("Memory Used (Bytes): ", mem_used)
        print("\n")

        writer.writerow([i, min_supp, exec_time, mem_used, len(rules)])
        
        gc.collect()

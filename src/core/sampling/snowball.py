import random


class Queue():
    # Constructor creates a list
    def __init__(self):
        self.queue = list()

    # Adding elements to queue
    def enqueue(self, data):
        if data not in self.queue:
            self.queue.insert(0, data)
            return True
        return False

    # Removing the last element from the queue
    def dequeue(self):
        if len(self.queue) > 0:
            return self.queue.pop()
        else:
            exit()

    # Getting the size of the queue
    def size(self):
        return len(self.queue)

    # printing the elements of the queue
    def printQueue(self):
        return self.queue


class Snowball():

    def __init__(self):
        self.nodes_list = []

    def snowball(self, G, size, k):
        q = Queue()
        m = k
        dictt = set()
        while(m):
            id = random.sample(list(G.nodes()), 1)[0]
            q.enqueue(id)
            m = m - 1
        while(len(self.nodes_list) <= size):
            if(q.size() > 0):
                id = q.dequeue()
                self.nodes_list.append(id)
                if(id not in dictt):
                    dictt.add(id)
                    list_neighbors = list(G.neighbors(id))
                    if(len(list_neighbors) > k):
                        for x in list_neighbors[:k]:
                            q.enqueue(x)
                            self.nodes_list.append(x)
                    elif(len(list_neighbors) <= k and len(list_neighbors) > 0):
                        for x in list_neighbors:
                            q.enqueue(x)
                            self.nodes_list.append(x)
                else:
                    continue
            else:
                initial_nodes = random.sample(
                    list(G.nodes()) and list(dictt), k)
                for id in initial_nodes:
                    q.enqueue(id)
        return G.subgraph(self.nodes_list)
        

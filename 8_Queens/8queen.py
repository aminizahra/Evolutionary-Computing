# In[1]:


import numpy as np
import pandas as pd


# In[2]:


def generate_population():
    chromosome = [[i, j, k, l, m, n, o, p]
           for i in range(1, 9)
           for j in range(1, 9)
           for k in range(1, 9)
           for l in range(1, 9)
           for m in range(1, 9)
           for n in range(1, 9)
           for o in range(1, 9)
           for p in range(1, 9)
           if all([i != j, i != k, i != l, i != m, i != n, i != o, i != p,
                   j != k, j != l, j != m, j != n, j != o, j != p,
                   k != l, k != m, k != n, k != o, k != p,
                   l != m, l != n, l != o, l != p,
                   m != n, m != o, m != p,
                   n != o, n != p,
                   o != p])]
    chromosome = np.array(chromosome)
    chromosome = pd.DataFrame(chromosome)
    return chromosome


# In[3]:


initial_population = generate_population()
initial_population


# In[4]:


def fitness(population):
    pop_size = population.shape[0]
    x = 0
    y = 0
    b = 0
    c = 0
    Fit = []
    for k in range(pop_size):
        for i in range(8):
            c = 0
            for j in range(8):
                if(i != j):
                    x = abs(i-j)
                    y = abs(population.iloc[k][i] - population.iloc[k][j])
                    if(x == y):
                        c += 1
        b = 28-c
        Fit.append(b)
    Fitness = np.array(Fit)
    return Fitness


# In[5]:


Fitness = fitness(initial_population)
Fitness


# In[6]:


data = pd.DataFrame(initial_population)
data['Fit'] = pd.DataFrame(Fitness)


# In[7]:


data_100 = data.sample(n=100)
data_100 = data_100.reset_index(drop = True)
data_100


# In[8]:


def selection(data):
    selected_parent = data.sample(n=5)
    selected_parent = selected_parent.sort_values("Fit", ascending=False)
    selected_parent1 = selected_parent.iloc[0]
    selected_parent2 = selected_parent.iloc[1] 
    return selected_parent1[:8], selected_parent2[:8]


# In[9]:


def crossover(C1, C2):
    point = np.random.randint((1,7), size=1)
    point = int(point)
    
    C1_1 = C1[:point]
    C1_2 = C1[point:]
    
    C2_1 = C2[:point]
    C2_2 = C2[point:]
    
    C1_tuple = (C1_1, C2_2)
    C1 = np.hstack(C1_tuple)

    C2_tuple = (C2_1, C1_2)
    C2 = np.hstack(C2_tuple)
    return C1, C2


# In[10]:


def mutation(ch):
    point1 = np.random.randint(8, size=1)
    point1 = int(point1)
    
    point2 = np.random.randint(8, size=1)
    point2 = int(point2)
    
    first_ele = ch[point1]  
    second_ele = ch[point2]
    
    ch[point1] = second_ele
    ch[point2] = first_ele
    
    return ch


# In[11]:


Parent1 = []
Child_Gen1 = []
for i in range(25):
    Pa1, Pa2 = selection(data_100)
    Parent1.append(Pa1)
    Parent1.append(Pa2)
    
    Child1, Child2 = crossover(Pa1, Pa2)

    Child1 = mutation(Child1)
    Child2 = mutation(Child2)
    
    Child_Gen1.append(Child1)
    Child_Gen1.append(Child2)
    
Parent1_df = pd.DataFrame(Parent1)
Parent1_df = Parent1_df.reset_index(drop = True)

Child_Gen1 = pd.DataFrame(Child_Gen1)
Child_Gen1 = Child_Gen1.reset_index(drop = True)


# In[12]:


Child_Gen1


# In[13]:


Gen1_Fitness = fitness(Child_Gen1)
data_Gen1 = Child_Gen1
data_Gen1['Fit'] = pd.DataFrame(Gen1_Fitness)


# In[14]:


data_Gen1


# In[15]:


Parent2 = []
Child_Gen2 = []
for i in range(12):
    Pa1, Pa2 = selection(data_Gen1)
    Parent2.append(Pa1)
    Parent2.append(Pa2)
    
    Child1, Child2 = crossover(Pa1, Pa2)

    Child1 = mutation(Child1)
    Child2 = mutation(Child2)
    
    Child_Gen2.append(Child1)
    Child_Gen2.append(Child2)
    
Parent2_df = pd.DataFrame(Parent2)
Parent2_df = Parent2_df.reset_index(drop = True)

Child_Gen2 = pd.DataFrame(Child_Gen2)
Child_Gen2 = Child_Gen2.reset_index(drop = True)


# In[16]:


Child_Gen2


# In[17]:


Gen2_Fitness = fitness(Child_Gen2)
data_Gen2 = Child_Gen2
data_Gen2['Fit'] = pd.DataFrame(Gen2_Fitness)


# In[18]:


data_Gen2


# In[19]:


Parent3 = []
Child_Gen3 = []
for i in range(6):
    Pa1, Pa2 = selection(data_Gen2)
    Parent3.append(Pa1)
    Parent3.append(Pa2)
    
    Child1, Child2 = crossover(Pa1, Pa2)

    Child1 = mutation(Child1)
    Child2 = mutation(Child2)
    
    Child_Gen3.append(Child1)
    Child_Gen3.append(Child2)
    
Parent3_df = pd.DataFrame(Parent3)
Parent3_df = Parent3_df.reset_index(drop = True)

Child_Gen3 = pd.DataFrame(Child_Gen3)
Child_Gen3 = Child_Gen3.reset_index(drop = True)


# In[20]:


Child_Gen3


# In[21]:


Gen3_Fitness = fitness(Child_Gen3)
data_Gen3 = Child_Gen3
data_Gen3['Fit'] = pd.DataFrame(Gen3_Fitness)


# In[22]:


data_Gen3


# In[23]:


Parent4 = []
Child_Gen4 = []
for i in range(4):
    Pa1, Pa2 = selection(data_Gen3)
    Parent4.append(Pa1)
    Parent4.append(Pa2)
    
    Child1, Child2 = crossover(Pa1, Pa2)

    Child1 = mutation(Child1)
    Child2 = mutation(Child2)
    
    Child_Gen4.append(Child1)
    Child_Gen4.append(Child2)
    
Parent4_df = pd.DataFrame(Parent4)
Parent4_df = Parent4_df.reset_index(drop = True)

Child_Gen4 = pd.DataFrame(Child_Gen4)
Child_Gen4 = Child_Gen4.reset_index(drop = True)


# In[24]:


Child_Gen4


# In[25]:


Gen4_Fitness = fitness(Child_Gen4)
data_Gen4 = Child_Gen4
data_Gen4['Fit'] = pd.DataFrame(Gen4_Fitness)


# In[26]:


data_Gen4


# In[27]:


CHILD = data_Gen1.append(data_Gen2)
CHILD = CHILD.append(data_Gen3)
CHILD = CHILD.append(data_Gen4)


# In[28]:


CHILD = CHILD.sort_values("Fit", ascending=False)
CHILD = CHILD.reset_index(drop = True)
CHILD.head(10)


# In[29]:


print("Finished!")


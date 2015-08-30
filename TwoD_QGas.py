import random

import scipy
import scipy.signal

import numpy.fft
import matplotlib.pyplot as plt

import numpy as np
from numpy import exp,arange


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


class Particle(object):
    
    def __init__(self,index,state,position):
        self.index = index
        self.state = state
        self.position = position
        self.pairing = None
        self.bell_state = None
        self.bell_state_time = None


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def _free_pairing(chosen, other, r=random.random()):

    chosen.pairing = other
    other.pairing = chosen

    chosen.state = 0
    other.state = 0
    chosen.bell_state_time = 0
    other.bell_state_time = 0

    if r<=0.25:
        chosen.bell_state = 1
        other.bell_state = 1
    elif r>0.25 and r<=0.5:
        chosen.bell_state = 2
        other.bell_state = 2
    elif r>0.5 and r<=0.75:
        chosen.bell_state = 3
        other.bell_state = 3
    elif r>0.75 and r<=1.0:
        chosen.bell_state = 4
        other.bell_state = 4

def _entangled_pairing(unpaired, paired, partner, r=random.random()):
    unpaired.pairing = paired
    paired.pairing = unpaired
    partner.pairing = None

    unpaired.state = 0
    paired.state = 0
    partner.bell_state = None
    unpaired.bell_state_time = 0
    paired.bell_state_time = 0
    partner.bell_state_time = 0

    if paired.bell_state in (1,2):
        flip = 1
    elif paired.bell_state in (3,4):
        flip = -1

    if r<=0.25:
        partner.state = flip*unpaired.state
        unpaired.bell_state = 1
        paired.bell_state =  1 
    elif r>0.25 and r<=0.5:
        partner.state = flip*unpaired.state
        unpaired.bell_state = 2
        paired.bell_state = 2
    elif r>0.5 and r<=0.75:
        partner.state = flip*(-1)*unpaired.state
        unpaired.bell_state = 3
        paired.bell_state = 3
    elif r>0.75:
        partner.state = flip*(-1)*unpaired.state
        unpaired.bell_state = 4
        paired.bell_state = 4

def _double_entangled_pairing(paireds, partners):
    r = random.random()
    _free_pairing(*paireds, r=r)
    _free_pairing(*partners, r=r)

def _break_entanglement(chosen, rule):
    s = (1 if rule() else -1)

    if chosen.bell_state in (1,2):
        chosen.state = s
        chosen.pairing.state = -s
    elif chosen.bell_state in (3,4):
        chosen.state = s
        chosen.pairing.state = s

    chosen.state = 1
    chosen.pairing.state = -1

    chosen.bell_state = None
    chosen.bell_state_time = None
    
    chosen.pairing.bell_state = None
    chosen.pairing.bell_state_time = None

    chosen.pairing.pairing=None
    chosen.pairing = None


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


class QuantumGas(object):

    def __init__(self, y, x, temp):
        self.temp = temp
        self.shape = (x,y)
        self.area = x*y
        space = np.ndenumerate(np.array([range(x),range(y)]))
        self.particles = [Particle(v,random.choice([-1,1]),i) for i,v in space]

    def __getitem__(self, key):
        return self.particles[key:(key+1)*self.shape[0]]
    
    @property
    def indexes(self):
        return np.array(p.index for p in self.particles)
    
    @property
    def pairings(self):
        return np.array(p.pairing for p in self.particles)
    
    @property
    def states(self):
        return np.array(p.state for p in self.particles)

    @property    
    def positions(self):
        return np.array(p.position for p in self.particles)

    @property    
    def bell_states(self):
        return np.array(p.bell_state for p in self.particles)
    
    @property
    def bell_state_times(self):
        return np.array(p.bell_state_times for p in self.particles)

    @property      
    def magnitization(self):
        net_moment = 0
        for i in range(self.shape[1]):
            for j in range(self.shape[0]):
                net_moment= net_moment+self[i][j].state
        return net_moment
    
    @property
    def weighted_magnitization(self):
        return self.magnitization/self.area
    @property
    def energy(self):
        M = self.weighted_magnitization
        #M=self.magnitization
        Energy = 0
        for i in range(self.shape[1]):
            for j in range(self.shape[0]):
                Energy = Energy+(M*self[i][j].state)
        
        return (-1*Energy)

    # can put back later...
    # def move(self, max_vel_temp):

    def _get_neighbor_indeces(self, i, j, directions='tblr'):
        """iterate over indeces of the neighbors i,j

        indeces given in counter clockwise order starting
        at the right neighbor. request specific neighbors
        by indicating the desired directions."""
        if 'r' in directions:
            if j==(self.shape[0]-1):
                yield (i,0)
            else:
                yield (i,j+1)

        if 't' in directions:
            if i==0:
                yield (self.shape[1]-1,j)
            else:
                yield (i-1,j)

        if 'l' in directions:
            if j==0:
                yield (i,self.shape[0]-1)
            else:
                yield (i,j-1)

        if 'b' in directions:
            if i==(self.shape[1]-1):
                yield (0,j)
            else:
                yield (i+1,j)

    def _nghbr_state_sum(self, i, j, directions='btlr'):
        indeces = self._get_neighbor_indeces(i, j, directions)
        nghbrs = [self[i][j] for i,j in indeces]
        return sum([n.state for n in nghbrs])

    @property    
    def ising_energy(self):
        E = 0
        for i in range(self.shape[1]):
            for j in range(self.shape[0]):
                if i==0 and j!=(self.shape[0]-1):
                    p1, p2, p3 = self[i][j], self[self.shape[1]-1][j], self[i][j+1]
                    E += (p1.state*p2.state) + (p1.state*p3.state)

                elif i==0 and j==(self.shape[0]-1):
                    p1, p2, p3 = self[i][j], self[self.shape[1]-1][j], self[i][0]
                    E += (p1.state*p2.state) + (p1.state*p3.state)

                elif i!=0 and j!=(self.shape[0]-1):
                    p1, p2, p3 = self[i][j], self[i-1][j], self[i][j+1]
                    E += (p1.state*p2.state) + (p1.state*p3.state)

                elif i!=0 and j==(self.shape[0]-1):
                    p1, p2, p3 = self[i][j], self[i-1][j], state*self[i][0]
                    E += (p1.state*p2.state) + (p1.state*p3.state)
        return -1*E

    def ising_interaction(self):
        T = float(self.temp)
        i = random.randint(0,self.shape[1]-1)
        j = random.randint(0,self.shape[0]-1)
        DE = self.ising_diffeq(i,j)
        prob = random.random()
        if DE<=0:
            self[i][j].state = self[i][j].state*(-1)
        elif prob<exp((-1*DE)/T):
            self[i][j].state = self[i][j].state*(-1)
    
    def ising_diffeq(self,i,j):
        state_sum = self._nghbr_state_sum(i,j)
        E_i = (-1*self[i][j].state)*(state_sum)
        E_f = self[i][j].state*(state_sum)
        return float(E_f-E_i)
    
    def ising_energy2(self):
        E = 0
        for i in range(self.shape[1]):
            for j in range(self.shape[0]):
                    state_sum = self._nghbr_state_sum(i,j)
                    E += (-1*self[i][j].state)*(state_sum)
        return float(E)

    def ising_bell_state_diffeq(self,i,j,k,p,direction):
        if direction=='top':
            state_sum1 = self._nghbr_state_sum(i,j,directions='blr')
            state_sum2 = self._nghbr_state_sum(k,p,directions='tlr')

            top = self[k][p].state
            de_pt1 = self[i][j].state*(top+state_sum1)
            de_pt2 = self[k][p].state*(state_sum2)
            DE = de_pt1 + de_pt2

        elif direction=='bottom':
            state_sum1 = self._nghbr_state_sum(i,j,directions='tlr')
            state_sum2 = self._nghbr_state_sum(k,p,directions='blr')

            bottom = self[k][p].state
            de_pt1 = self[i][j].state*(bottom+state_sum1)
            de_pt2 = self[k][p].state*(state_sum2)
            DE = de_pt1 + de_pt2

        elif direction=='left':
            state_sum1 = self._nghbr_state_sum(i,j,directions='tbr')
            state_sum2 = self._nghbr_state_sum(k,p,directions='tbl')

            left = self[k][p].state
            de_pt1 = self[i][j].state*(left+state_sum1)
            de_pt2 = self[k][p].state*(state_sum2)
            DE = de_pt1 + de_pt2

        elif direction=='right':
            state_sum1 = self._nghbr_state_sum(i,j,directions='tbl')
            state_sum2 = self._nghbr_state_sum(k,p,directions='tbr')

            right = self[k][p].state
            de_pt1 = self[i][j].state*(right+state_sum1)
            de_pt2 = self[k][p].state*(state_sum2)
            DE = de_pt1 + de_pt2

        return DE

    def bell_state_diffeq(self,i,j,k,p):
        M = 0
        for s in range(self.shape[1]):
            for t in range(self.shape[0]):
                if (s!=i or t!=j) and (s!=k or t!=p):
                    M = M+self[s][t].state
        WM = M/self.area

        E_f = 0
        for s in range(self.shape[1]):
            for t in range(self.shape[0]):
                if (s!=i or t!=j) and (s!=k or t!=p):
                    E_f += (WM*self[s][t].state)
        E_f = (-1)*E_f

        E_i = self.energy
        return E_f-E_i

    def bell_state_interaction(self, ising_energy=None):
        i = random.randint(0,self.shape[1]-1)
        j = random.randint(0,self.shape[0]-1)

        indeces = list(self._get_neighbor_indeces(i,j))
        choice_list = ['right','top','left','bottom']

        c = random.choice((0,1,2,3))
        direciton = choice_list[c]
        k,p = indeces[c]

        if ising_energy=='Use Ising Energy':
            DE = self.ising_bell_state_diffeq(i,j,k,p,direction)
            #print('Using Ising Energy')
        else:
            DE = self.bell_state_diffeq(i,j,k,p)
        T = float(self.temp)
        chosen = self[i][j]
        other = self[k][p]
        
        if not chosen.pairing and not other.pairing:
            if DE<=0:
                _free_pairing(chosen, other)
            elif random.random()< np.exp((-1*DE)/T):
                _free_pairing(chosen, other)
        
        elif bool(chosen.pairing) != bool(other.pairing):
            partner = chosen.pairing or other.pairing
            r = random.random()
            if DE<=0:
                ps = (chosen, other, partner)
                _entangled_pairing(*ps)
            elif random.random()<np.exp((-1*DE)/T):
                ps = (chosen, other, partner)
                _entangled_pairing(*ps)
                    
        elif chosen.pairing and other.pairing:
            r = random.random()
            if DE<=0:
                paireds = (other, chosen)
                partners = (chosen.pairing, other.pairing)
                _double_entangled_pairing(paireds, partners)
            elif random.random()<np.exp((-1*DE)/T):
                paireds = (other, chosen)
                partners = (chosen.pairing, other.pairing)
                _double_entangled_pairing(paireds, partners)

    def entanglement_density(self):
        count = 0
        for i in range(self.shape[1]):
            for j in range(self.shape[0]):
                if self[i][j].pairing!=None:
                    count = count+1
                else:
                    count = count
        return float(count)/float(self.area)
    
    def decoherence(self, tau):
        rule = lambda : random.random()<=0.5
        for i in range(self.shape[1]):
            for j in range(self.shape[0]):
                if self[i][j].pairing:
                    r = random.random()
                    if r<(1.-exp((-1.*self[i][j].bell_state_time)/tau)):
                        _break_entanglement(self[i][j], rule)
                if self[i][j].pairing:
                    self[i][j].bell_state_time+=1

    def _energy_definition(self, interaction_type):
        if interaction_type in ('Ising','Ising Bell State','Ising and Bell State'):
            return self._state_sum_energy
        elif interaction_type in ('Baseline','Bell State','Bell and Baseline':
            return _magnitization_energy


    def _state_sum_energy(self, i, j):
        state_sum = self._nghbr_state_sum(i,j)
        E_down = state_sum
        E_up = -state_sum
        return E_up, E_down

    def _magnitization_energy(self, i, j):
        """ignores i, j"""
        M=self.magnitization
        M_up = M+1
        M_down = M-1
        E_up = M_up
        E_down = -M_down
        return E_up, E_down

    def decoherence_2(self, tau, interaction_type):
        energy_calculator = self._energy_definition(interaction_type)

        decohere_by_random = lambda : random.random()<=0.5

        class decohere_by_energy(object):
            def __init__(E_up,E_down):
                self.E_up = E_up
                self.E_down = E_down
            def __call__():
                return self.E_up<self.E_down
        
        for i in range(self.shape[1]):
            for j in range(self.shape[0]):
                if self[i][j].pairing:
                    r=random.random()
                    if r<(1.-exp((-1.*self[i][j].bell_state_time)/tau)):

                        E_up, E_down = energy_calculator(i,j)
                        if E_up!=E_down:
                            rule = decohere_by_energy(E_up,E_down)
                        else:
                            rule = decohere_by_random
                        _break_entanglement(self[i][j], rule)
                if self[i][j].pairing:
                    self[i][j].bell_state_time+=1


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


import sys

class ProgressBar(object):
    
    def __init__(self, length=20):
        self.length = length
        self.current = None
        
    def printout(self, percent):
        bar = "="*int(round(percent*self.length))
        if self.current!=bar:
            form = "\r[%-"+str(self.length)+"s]"
            sys.stdout.write(form % bar)
            sys.stdout.flush()
            self.current = bar


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def _seed_adjacent_bell_states(number, g):
    for k in range(number):
        seeded=False
        while not seeded:
            i = random.randint(0,g.shape[1]-1)
            j = random.randint(0,g.shape[0]-1)

            chosen = g[i][j]
            if chosen.state is None
                indeces = list(g._get_neighbor_indeces(i,j))
                c = random.choice((0,1,2,3))
                k,p = indeces[c]

                other = g[k][p]
                if other.state is None:
                    _free_pairing(chosen, other)
                    seeded = True

def _seed_seperated_bell_states(number, g):
    for k in range(number_seeded_bell_states):
        seeded=False
        while not seeded:
            i = random.randint(0,g.shape[1]-1)
            j = random.randint(0,g.shape[0]-1)

            chosen = g[i][j]
            if chosen.state is None
                other=None
                while not other and other.state is None:
                    k = random.randint(0,g.shape[1]-1)
                    p = random.randint(0,g.shape[0]-1)
                    if k!=i or p!=j:
                        other = g[k][p]

                _free_pairing(chosen,other)
                seeded = True

def iterate(N, max_vel_temp, tau, interaction, movement=None, decoherence='Yes',
            decoherence_type=2, input_gas=None, gas_shape=None, temperature=None,
            seed_bell_states=None, number_seeded_bell_states=None):
    #Iteration function. Parameters:
    #N is number of iterations
    #gas_shape is shape of gas given in tuple [y_length, x_length]
    #Temperatrue is temp of gas
    #max_vel_temp is used in movement function
    #tau is used in decoherence function
    #Interaction indicates what interaction, given as string. Interactions are, Ising, Ising Bell State, Baseline, Bell State
    #input_gas is gas input into function if the function makes no gas. Defalt is None so that function will make own gas
    I=np.zeros(N)
    E=np.zeros(N)
    E_2=np.zeros(N)
    M=np.zeros(N)
    M_2=np.zeros(N)
    ED=np.zeros(N)

    if input_gas==None:
        g=QuantumGas(gas_shape[0],gas_shape[1], temperature)

    if seed_bell_states=='adjacent':
        _seed_adjacent_bell_states(number_seeded_bell_states, g)

    elif seed_bell_states=='separated':
        _seed_seperated_bell_states(number_seeded_bell_states, g)

    for i in range(N):
        if interaction=='Ising':
            g.ising_interaction()
            if movement=='yes':
                g.move(max_vel_temp)
        elif interaction=='Ising Bell State':
            g.bell_state_interaction('Use Ising Energy')
            if decoherence=='Yes':
                if decoherence_type==1:
                    g.decoherence(tau)
                elif decoherence_type==2:
                    g.decoherence_2(tau, interaction)
            if movement=='yes':
                g.move(max_vel_temp)
        elif interaction=='Ising and Bell State':
            g.bell_state_interaction('Use Ising Energy')
            g.ising_interaction()
            if decoherence=='Yes':
                if decoherence_type==1:
                    g.decoherence(tau)
                elif decoherence_type==2:
                    g.decoherence_2(tau,interaction)
            if movement=='yes':
                g.move(max_vel_temp)
        elif interaction=='Baseline':
            g.baseline_interaction()
            if decoherence=='Yes':
                if decoherence_type==1:
                    g.decoherence(tau)
                elif decoherence_type==2:
                    g.decoherence_2(tau, interaction)
            if movement=='yes':
                g.move(max_vel_temp)
        elif interaction=='Bell State':
            g.bell_state_interaction()
            if decoherence=='Yes':
                if decoherence_type==1:
                    g.decoherence(tau)
                elif decoherence_type==2:
                    g.decoherence_2(tau,interaction)
            if movement=='yes':
                g.move(max_vel_temp)
        elif interaction=='Bell and Baseline':
            g.bell_state_interaction()
            g.baseline_interaction()
            if decoherence=='Yes':
                if decoherence_type==1:
                    g.decoherence(tau)
                elif Dechoerence_type==2:
                    g.decoherence_2(tau,interaction)
            if movement=='yes':
                g.move(max_vel_temp)
        I[i]=i
        if interaction in ('Ising','Ising Bell State','Ising and Bell State'):
            E[i] = g.ising_energy
            E_2[i] = (g.ising_energy)**2.
        else:
            E[i] = g.energy
            E_2[i] = (g.energy)**2.
        M[i] = g.magnitization
        M_2[i] = (g.magnitization)**2.
        ED[i] = g.entanglement_density()
    
    return g, I, E, E_2, M, M_2, ED


def temperature_iteration(T_i, T_f, N_Temps, N, max_vel_temp, tau, interaction, movement=None,
                        decoherence='Yes', decoherence_type=2, gas_in=None, gas_shape=None,
                        temperature=None, seed_bell_states=None, number_seeded_bell_states=None,
                        extra_iteration=None, extra_iteration_num=None):
    #Temperature iteration function
    # T_i=inital temperature, T_f=final temperature, N_Temps is number if temperatures between T_i and T_f
    # all other arguments are the relevent arguments in the Iterate function
    T=np.linspace(T_i,T_f,N_Temps)
    E=np.zeros(len(T))
    CV=np.zeros(len(T))
    M=np.zeros(len(T))
    Chi=np.zeros(len(T))
    ED=np.zeros(len(T))

    if gas_in is None:
        g = QuantumGas(gas_shape[0],gas_shape[1],temperature)

    if seed_bell_states=='adjacent':
        _seed_adjacent_bell_states(number_seeded_bell_states, g)

    elif seed_bell_states=='separated':
        _seed_seperated_bell_states(number_seeded_bell_states, g)

    pb = ProgressBar(20)
    for i in range(len(T)):
        g.temp=T[i]

        if extra_iteration=='Yes':
            args = (extra_iteration_num,
                    max_vel_temp,
                    tau,interaction)
            g,a,b,c,d,e,f = iterate(*args, input_gas=g)

        args = (N,max_vel_temp,tau,interaction)
        kwargs = {'movement':movement,
                  'decoherence':decoherence,
                  'decoherence_type':decoherence_type,
                  'input_gas':g}

        g,I,EI,E_2I,MI,M_2I,EDI = Iterate(*args,**kwargs)
        E[i] = np.average(EI)
        CV[i] = (1./((float(T[i]))**2.))*(np.average(E_2I)-((np.average(EI))**2.))
        M[i]=np.average(MI)
        Chi[i]=(1./float(T[i]))*(np.average(M_2I)-((np.average(MI))**2.))
        ED[i]=np.average(EDI)
        pb.printout(float(i)/len(T))
        
    return g, T, E, CV, M, Chi, ED


# In[9]:

def sym_derivative(x,y,smooth=None):
    #if smooth=='yes':
     #   y_in=scipy.signal.savgol_filter(y,51,3)
    #elif smooth=='no':
      #  y_in=y
    deriv=np.zeros(len(x))
    for i in range(len(x)):
        if i==0:
            dy=y[i+1]-y[i]
            dx=x[i+1]-x[i]
            deriv[i]=dy/dx
        elif i==len(x)-1:
            dy=y[i]-y[i-1]
            dx=x[i]-x[i-1]
            deriv[i]=dy/dx
        else:
            dy=y[i+1]-y[i-1]
            dx=x[i+1]-x[i-1]
            deriv[i]=dy/dx
    return deriv
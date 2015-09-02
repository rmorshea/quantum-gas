import sys

import random

import scipy
import scipy.signal

import numpy.fft

import numpy as np
from numpy import exp,arange


# - - - - - - - - - - - - - - - - - - - - - - - - - - - -


class Particle(object):
    
    def __init__(self, state):
        self.state = state
        self.pairing = None
        self.bell_state = None
        self.bell_state_times = None
        self.limbdo = False


# - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# General Utilities - - - - - - - - - - - - - - - - - - -

def binit(value, bins):
    """return the bin index of the value"""
    for i in range(len(bins)):
        if value>bins[i]:
            return i-1
    return i

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

class radial_gridspace_factory(type):
    
    _orthogonals = ((1,1),(-1,1),(-1,-1),(1,-1))

    @classmethod
    def _gen_orths(cls, i, j):
        return [(i*o[0],j*o[1]) for o in cls._orthogonals]
    
    def __new__(meta, *space):
        if len(space)>2:
            raise ValueError('only supports 2D spaces')
        grid_shape = range(space[0]),range(space[1])
        x,y = np.meshgrid(*grid_shape,sparse=True)

        grid = x**2+y**2
        indices = {0:[(0,0)]}
        for j in range(1,grid.shape[1]):
            for i in range(0,grid.shape[0]):
                if indices.get(grid[i][j],None):
                    for n,m in meta._gen_orths(i,j):
                        indices[grid[i][j]].append((n,m))
                else:
                    indices[grid[i][j]] = meta._gen_orths(i,j)
        rads = sorted(indices.keys())
        return type('radial_gridspace',
                    (radial_gridspace_template,object),
                    {'indices':indices,'rads':rads})
        
class radial_gridspace_template:
    
    def __init__(self, *origin):
        if len(origin)>2:
            raise ValueError('only supports 2D indices')
        if len(origin)==0:
            origin = (0,0)
        self.origin = origin
        
    def __iter__(self):
        self.i = 0
        return self

    def __getitem__(self, index):
        o = self.origin
        r = self.rads[index]
        g = self.indices[r]
        return [(p[0]-o[0],p[1]-o[1]) for p in g]
    
    def next(self):
        try:
            r = self.rads[self.i]
        except IndexError:
            raise StopIteration
        self.i += 1
        return self.indices[r]

class EmptyGas(object):

    def __init__(self, x, y):
        self.particles = tuple(None for n in range(x*y))

    def __getitem__(self, key):
        return self.particles[key:(key+1)*self.shape[0]]


# - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Entanglement Utilities  - - - - - - - - - - - - - - - -

def _free_pairing(chosen, other):
    r=random.random()

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

def _entangled_pairing(unpaired, paired, partner):
    r=random.random()

    paired.state = 0
    paired.bell_state_time = 0

    partner.bell_state = None
    partner.bell_state_time = 0

    unpaired.state = 0
    unpaired.bell_state_time = 0

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

    unpaired.pairing = paired
    paired.pairing = unpaired
    partner.pairing = None

def _double_entangled_pairing(paireds, partners):
    _free_pairing(*paireds)
    _free_pairing(*partners)

def _break_entanglement(chosen, new_state):

    if chosen.bell_state in (1,2):
        chosen.state = new_state
        chosen.pairing.state = -new_state
    elif chosen.bell_state in (3,4):
        chosen.state = new_state
        chosen.pairing.state = new_state

    chosen.bell_state = None
    chosen.bell_state_time = None
    
    chosen.pairing.bell_state = None
    chosen.pairing.bell_state_time = None

    chosen.pairing.pairing=None
    chosen.pairing = None


# - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# QuantumGas Class  - - - - - - - - - - - - - - - - - - -


class QuantumGas(object):

    def __init__(self, x, y, temp):
        self.temp = temp
        self.shape = (x,y)
        self.area = x*y
        state_gen = lambda : random.choice((-1,1))
        self.particles = tuple(Particle(state_gen()) for i in range(x*y))
        # high upfront computation cost, but essentailly none afterwards.
        self.radial_gridspace_type = radial_gridspace_factory(x,y)

    def __getitem__(self, key):
        return self.particles[key:(key+1)*self.shape[0]]
    
    @property
    def pairings(self):
        return np.array([p.pairing for p in self.particles])
    
    @property
    def states(self):
        return np.array([p.state for p in self.particles])

    @property    
    def bell_states(self):
        return np.array([p.bell_state for p in self.particles])
    
    @property
    def bell_state_times(self):
        return np.array([p.bell_state_times for p in self.particles])

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

    def move(self, max_vel_temp):
        xlim, ylim = self.shape
        gas_template = EmptyGas(*self.shape)

        for i in range(self.shape[1]):
            for j in range(self.shape[0]):
                radial_gridspace = self.radial_gridspace_type(0,0)
                gauss_mu = self.shape[1]*(1.-exp(-self.temp/max_vel_temp))
                gauss_sigma = self.shape[1]*(1.-exp(-self.temp/max_vel_temp))/2.
                gnum = abs(random.gauss(guass_mu, gauss_sigma))
                # velocity is binned into a radius groups, based
                # on the allowed distances in `radial_gridspace`
                velocity = binit(gnum, griderator.rads.keys())
                
                if velocity>0:
                    # velocity now corrispods to a particular
                    # set of points with identical radii
                    rgroup = radial_gridspace[velocity]
                    diff = random.choice(rgroup)
                    
                    direction = random.random()
                    if direction<=0.5:
                        #print(velocity)
                        
                        if direction > 0.25 and direction <= 0.5:
                            if velocity > (self.x_length()-j-1):
                                for m in range(velocity-(self.x_length()-j),-1,-1):
                                    if new[i][m].index=='empty':
                                        new[i][m]=self[i][j]
                                        moved='yes'
                                        break
                                if moved=='no':
                                    for m in range(self.x_length()-1,j-1,-1):
                                        if new[i][m].index=='empty':
                                            new[i][m]=self[i][j]
                                            moved='yes'
                                            break
                                    if moved=='no':
                                        for k in range(self.y_length()):
                                            for p in range(self.x_length()):
                                                if new[k][p].index=='empty':
                                                    new[k][p]=self[i][j]
                                                    moved='yes'
                                                    break
                                            if moved=='yes':
                                                break
                                    
                            else:
                                for m in range(j+velocity,j-1,-1):
                                    if new[i][m].index=='empty':
                                        new[i][m]=self[i][j]
                                        moved='yes'
                                        break
                        
                                if moved=='no':
                                    for k in range(self.y_length()):
                                        for p in range(self.x_length()):
                                            if new[k][p].index=='empty':
                                                new[k][p]=self[i][j]
                                                moved='yes'
                                                break
                                        if moved=='yes':
                                            break
                                            
                        elif direction <= 0.25:
                            if velocity > j:
                                for m in range(self.x_length()-(velocity-j),self.x_length()):
                                    if new[i][m].index=='empty':
                                        new[i][m]=self[i][j]
                                        moved='yes'
                                        break
                                if moved=='no':
                                    for m in range(0,j+1):
                                        if new[i][m].index=='empty':
                                            new[i][m]=self[i][j]
                                            moved='yes'
                                            break
                                    if moved=='no':
                                        for k in range(self.y_length()):
                                            for p in range(self.x_length()):
                                                if new[k][p].index=='empty':
                                                    new[k][p]=self[i][j]
                                                    moved='yes'
                                                    break
                                            if moved=='yes':
                                                break
                                    
                            else:
                                for m in range(j-velocity,j+1):
                                    if new[i][m].index=='empty':
                                        new[i][m]=self[i][j]
                                        moved='yes'
                                        break
                                if moved=='no':
                                    for k in range(self.y_length()):
                                        for p in range(self.x_length()):
                                            if new[k][p].index=='empty':
                                                new[k][p]=self[i][j]
                                                moved='yes'
                                                break
                                        if moved=='yes':
                                            break
                                                
                    elif direction > 0.5:
                        if velocity > vmax_y:
                            velocity = vmax_y
                        #print(velocity)
                        
                        if direction > 0.75:
                            if velocity > (self.y_length()-i-1):
                                for m in range(velocity-(self.y_length()-i),-1,-1):
                                    if new[m][j].index=='empty':
                                        new[m][j]=self[i][j]
                                        moved='yes'
                                        break
                                if moved=='no':
                                    for m in range(self.y_length()-1,i-1,-1):
                                        if new[m][j].index=='empty':
                                            new[m][j]=self[i][j]
                                            moved='yes'
                                            break
                                    if moved=='no':
                                        for k in range(self.y_length()):
                                            for p in range(self.x_length()):
                                                if new[k][p].index=='empty':
                                                    new[k][p]=self[i][j]
                                                    moved='yes'
                                                    break
                                            if moved=='yes':
                                                break
                                    
                            else:
                                for m in range(i+velocity,i,-1):
                                    if new[m][j].index=='empty':
                                        new[m][j]=self[i][j]
                                        moved='yes'
                                        break
                        
                                if moved=='no':
                                    for k in range(self.y_length()):
                                        for p in range(self.x_length()):
                                            if new[k][p].index=='empty':
                                                new[k][p]=self[i][j]
                                                moved='yes'
                                                break
                                        if moved=='yes':
                                            break
                                                
                        elif direction > 0.5 and direction <=0.75:
                            if velocity > i:
                                for m in range(self.y_length()-(velocity-i),self.y_length()+1):
                                    if new[m][j].index=='empty':
                                        new[m][j]=self[i][j]
                                        moved='yes'
                                        break
                                if moved=='no':
                                    for m in range(0,i+1):
                                        if new[m][j].index=='empty':
                                            new[m][j]=self[i][j]
                                            moved='yes'
                                            break
                                    if moved=='no':
                                        for k in range(self.y_length()):
                                            for p in range(self.x_length()):
                                                if new[k][p].index=='empty':
                                                    new[k][p]=self[i][j]
                                                    moved='yes'
                                                    break
                                            if moved=='yes':
                                                break
                                    
                            else:
                                for m in range(i-velocity,i+1):
                                    if new[m][j].index=='empty':
                                        new[m][j]=self[i][j]
                                        moved='yes'
                                        break
                                if moved=='no':
                                    for k in range(self.y_length()):
                                        for p in range(self.x_length()):
                                            if new[k][p].index=='empty':
                                                new[k][p]=self[i][j]
                                                moved='yes'
                                                break
                                        if moved=='yes':
                                            break

                moved = False
                griderator.origin = (i,j)
                for rgroup in griderator:
                    for n,m in random.shuffle(rgroup):
                        if 0<n<self.shape[0] and 0<m<self.shape[1]:
                            if gas_template[n][m] is None:
                                gas_template[n][m] = self[i][j]
                                moved = True
                                break
                    if moved:
                        break
                                        
        self.particles = gas_template.particles

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
                    p1, p2, p3 = self[i][j], self[i-1][j], self[i][0]
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
        direction = choice_list[c]
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

            if chosen.pairing:
                paired = chosen
                unpaired = other
            else:
                paired = other
                unpaired = chosen

            r = random.random()
            if DE<=0:
                ps = (unpaired, paired, partner)
                _entangled_pairing(*ps)
            elif random.random()<np.exp((-1*DE)/T):
                ps = (unpaired, paired, partner)
                _entangled_pairing(*ps)

        else:
            r = random.random()
            if DE!=0:
                raise ValueError('DE should always be 0')
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
        elif interaction_type in ('Baseline','Bell State','Bell and Baseline'):
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
        
        for i in range(self.shape[1]):
            for j in range(self.shape[0]):
                if self[i][j].pairing:
                    r=random.random()
                    if r<(1.-exp((-1.*self[i][j].bell_state_time)/tau)):

                        E_up, E_down = energy_calculator(i,j)
                        if E_up!=E_down:
                            new_state = 1 if E_up<E_down else -1
                        else:
                            new_state = 1 if random.random()<=0.5 else -1
                        _break_entanglement(self[i][j], new_state)
                if self[i][j].pairing:
                    self[i][j].bell_state_time+=1


# - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Iterator Function Utilities - - - - - - - - - - - - - -


def _seed_adjacent_bell_states(number, g):
    for k in range(number):
        seeded=False
        while not seeded:
            i = random.randint(0,g.shape[1]-1)
            j = random.randint(0,g.shape[0]-1)

            chosen = g[i][j]
            if chosen.state is None:
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
            if chosen.state is None:
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
        g = QuantumGas(gas_shape[0],gas_shape[1], temperature)
    else:
        g = input_gas

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

    pb = ProgressBar(50)
    pb.printout(0)
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

        g,I,EI,E_2I,MI,M_2I,EDI = iterate(*args,**kwargs)
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
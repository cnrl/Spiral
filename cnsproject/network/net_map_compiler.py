import re
import numpy as np
from ..utils import Container

class NetworkCompiler:
    def __init__(self):
        self.constructors = {}
        self.INTERSECTION = ['+']
        self.HORIZONTAL = ['-']
        self.VERTICAL = ['|']
        self.VERTICAL_WIRE = self.INTERSECTION+self.VERTICAL
        self.HORIZONTAL_WIRE = self.INTERSECTION+self.HORIZONTAL
        self.LEFT = ['<']
        self.RIGHT = ['>']
        self.DOWN = ['v']
        self.UP = ['^']
        self.HORIZONTAL_WAY = self.HORIZONTAL_WIRE+self.LEFT+self.RIGHT
        self.VERTICAL_WAY = self.VERTICAL_WIRE+self.DOWN+self.UP
        self.POPULATION_LEFT = ['[']
        self.POPULATION_RIGHT = [']']
        self.POPULATION_GARD = self.POPULATION_LEFT+self.POPULATION_RIGHT
        self.NAME_SYMBOL = ['/']
        self.LEARNER_GARD = ['$']
        
    def build_net_map(self, net_str):
        lines = net_str.split('\n')
        w = max([len(r) for r in lines])
        for i in range(len(lines)):
            lines[i] = ' '+lines[i]+' '*(w-len(lines[i]))+' '
        lines = [' '*(w+2)]+lines+[' '*(w+2)]
        net_map = [list(r) for r in lines]
        self.lines = lines
        self.net_map = np.array(net_map)
        self.region_map = np.full_like(net_map, '').tolist()
        
    def set_region(self,y,x,name):
        if self.region_map[y][x] != '':
            assert self.region_map[y][x]==name, "Tow regions have same pixel!!"
            return
        self.region_map[y][x] = name
        
        here = self.net_map[y,x]
        if here in self.HORIZONTAL_WIRE:
            for i in [-1,+1]:
                there = self.net_map[y,x+i]
                if there in self.HORIZONTAL_WAY:
                    self.set_region(y,x+i,name)
                elif here not in self.INTERSECTION\
                and there in self.VERTICAL\
                and self.net_map[y,x+2*i] in self.HORIZONTAL:
                    self.set_region(y,x+2*i,name)
                    
        if here in self.VERTICAL_WAY:
            for i in [-1,+1]:
                there = self.net_map[y+i,x]
                if there in self.VERTICAL_WIRE:
                    self.set_region(y+i,x,name)
                elif here not in self.INTERSECTION\
                and there in self.HORIZONTAL\
                and self.net_map[y+2*i,x] in self.VERTICAL:
                    self.set_region(y+2*i,x,name)
                    
        return True
    
    def build_object(self, definition, extra_inputs=[]):
        if definition=='':
            return None
        constructor = definition.split('(')[0]
        inputs = re.search(r'\((.*)\)', definition).group(1)
        inputs = re.split(",(?=(?:[^\(]*\([^\)]*\))*[^\)]*$)", inputs)
        inputs = [eval(i) for i in inputs if i!='']
        print(f"{constructor}({*extra_inputs, *inputs})")
        return self.constructors[constructor](*extra_inputs, *inputs)
    
    def set_population_region(self, y, x, name):
        for i in [-1,len(re.split("|".join(self.POPULATION_RIGHT), self.lines[y][x:])[0])+1]:
            if self.net_map[y,x+i] in self.HORIZONTAL_WAY:
                self.set_region(y,x+i,name)
    
    def build_populations(self):
        y,x = np.where(np.isin(self.net_map,self.POPULATION_LEFT))
        population_locations = list(zip(y.tolist(),x.tolist()))
        for y,x in population_locations:
            definition = re.split("|".join(self.POPULATION_RIGHT), self.lines[y][x+1:])[0]
            definition,name = re.split("|".join(self.NAME_SYMBOL), definition)
            self.populations[name] = self.build_object(definition)
            self.set_population_region(y,x,name)
    
    def set_synapses_region(self, y, x, name):
        self.region_map[y][x] = name
        for i in [-1,1]:
            if self.net_map[y+i,x] in self.UP+self.DOWN:
                self.set_region(y+i,x,name)
                                       
    def build_synapses(self, direction='r'):
        direction = 1 if direction=='r' else -1
        direction_sym = self.RIGHT if direction==1 else self.LEFT
        y,x = np.where(np.isin(self.net_map, direction_sym))
        synapse_locations = list(zip(y.tolist(),x.tolist()))
        for y,x in synapse_locations:
            before = self.net_map[y,x-direction]
            if before not in self.HORIZONTAL_WIRE+self.POPULATION_GARD:
                continue
            presyn = self.region_map[y][x]
            line = self.lines[y][x+1:] if direction==1 else self.lines[y][:x]
            elements = re.split("|".join(direction_sym), line)
            if direction==-1: elements = list(reversed(elements))
            axon_def, synapse_def, dendrite_def, _ = elements
            postsyn = self.region_map[y][x+direction*(len(axon_def)+1+len(synapse_def)+1+len(dendrite_def)+1)]
            name = presyn+'_'+postsyn
            presyn = self.populations[presyn]
            postsyn = self.populations[postsyn]
            axon = self.build_object(axon_def, [presyn.shape])
            if axon is None: axon = Container({
                'shape': presyn.shape,
                'population_shape': presyn.shape,
                'terminal_shape': ()
            })
            dendrite = self.build_object(dendrite_def, [axon.shape, postsyn.shape])
            if dendrite is None: dendrite = Container({
                'shape': (*axon.shape, *postsyn.shape),
                'terminal_shape': axon.shape,
                'population_shape': postsyn.shape,
            })
            synapse = self.build_object(synapse_def, [axon, dendrite])
            self.synapses[name] = synapse
            presyn.add_axon_set(synapse.axon_set, name)
            postsyn.add_dendrite_sets(synapse.dendrite_set, name)
            synapse_x = x + (len(axon_def) if direction==1 else len(dendrite_def)) + 2
            self.set_synapses_region(y, synapse_x, name)
    
    def build_learners(self):
        y,x = np.where(np.isin(self.net_map,self.LEARNER_GARD))
        learner_locations = list(zip(y.tolist(),x.tolist()))
        for y,x in learner_locations:
            syn = None
            for i in [-1,1]:
                syn = self.region_map[y+i][x+1]
                if syn in self.synapses:
                    break
            else:
                continue
            definition = re.split("|".join(self.LEARNER_GARD), self.lines[y][x+1:])[0]
            self.learners[syn+'_LR'] = self.build_object(definition, [self.synapses[syn]])
                                
    def build_net(self):
        for k,v in self.populations.items():
            self.net.add_population(v, k)
        for k,v in self.synapses.items():
            self.net.add_synapse(v, k)
        for k,v in self.learners.items():
            self.net.add_LR(v, k)
    
    def compile(self, net, net_str):
        self.net = net
        self.build_net_map(net_str)
        self.populations = {}
        self.build_populations()
        self.synapses = {}
        self.build_synapses('r')
        self.build_synapses('l')
        self.learners = {}
        self.build_learners()
        self.build_net()
        return self.net
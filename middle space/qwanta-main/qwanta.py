import simpy
import networkx as nx
from qubit.qubit import PhysicalQubit
import os 
import pandas as pd
import seaborn as sns
import pickle
import dill
import ast
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

from QuantumProcess import _EntanglementPurification, _EntanglementSwapping, _GenerateLogicalResource, _GeneratePhyscialResource, _VirtualStateTomography
from SubProcess import _TimeLag
from Tomography.Tomography_physical import Tomography

class Experiment:
    def __init__(self, topologies, timelines, nodes_info=None, memFunc=None, gateError=None, simTime=None, measurementError=None,
                 parameters_set=None, label_records=None, BSA_prob=None,
                 path='experiment', message_log='exp', repeat=5,
                 progress_bar=False, collect_fidelity_history=False) -> None:

        self.experiments = list(timelines.keys())
        self.topologies = {}
        self.timelines = {}
        self.memFunc = {}
        self.gateError = {}
        self.measurementError = {}
        self.simTime = {}
        self.nodes_info = {}
        self.label_recorded = {}
        self.progress_bar = not progress_bar

        if label_records is None:
            label_records = {exper: 'Physical' for exper in self.experiments}

        for exper in self.experiments:
            self.topologies[exper] = [topologies[exper] for _ in range(len(parameters_set))]
            self.timelines[exper] = [timelines[exper] for _ in range(len(parameters_set))]
            self.memFunc[exper] = [memFunc[exper] for _ in range(len(parameters_set))]
            self.gateError[exper] = [gateError[exper] for _ in range(len(parameters_set))]
            self.measurementError[exper] = [measurementError[exper] for _ in range(len(parameters_set))]
            self.simTime[exper] = [simTime[exper] for _ in range(len(parameters_set))]
            self.label_recorded[exper] = label_records[exper]

            if nodes_info is not None:
                self.nodes_info[exper] = [nodes_info[exper] for _ in range(len(parameters_set))]

        if BSA_prob is None:
            BSA_prob = [1]*len(parameters_set)

        self.parameters_set = parameters_set
        self.BSA_prob = BSA_prob
        self.path = path
        self.message_log = message_log
        self.repeat = repeat
        self.configurations = {}
        self.executed = False
        self.Fidelity = {}
        self.Time_used = {} # Simulation time used to complete unlimited process
        
        for exper in self.experiments:
            self.configurations[exper] = []
            for index in range(len(parameters_set)):
                tmp_list = []
                for rd in range(repeat):
                    config = Configuration(self.topologies[exper][index], self.timelines[exper][index],
                                           nodes_info=self.nodes_info[exper][index],
                                           memFunc=self.memFunc[exper][index],
                                           gate_error=self.gateError[exper][index],
                                           measurementError=self.measurementError[exper][index],
                                           sim_time=self.simTime[exper][index],
                                           result_path=self.path,
                                           message=f'exp{exper}_p{self.parameters_set[index]}_r{rd}',
                                           collectFidelityHistory=collect_fidelity_history,
                                           experiment=self.message_log, label_record=self.label_recorded[exper])
                    tmp_list.append(config)
                self.configurations[exper].append(tmp_list)
                
    def run(self, save_tomo=False, save_result=True):

        for exper in tqdm(self.experiments, desc='Experiments executed: ',disable=self.progress_bar):
            self.Fidelity[exper] = []
            self.Time_used[exper] = []
            for index in tqdm(range(len(self.parameters_set)), desc='Parameters executed: ',disable=self.progress_bar, leave=False):
                tmp_list_fidelity = []
                tmp_list_time_used = []
                for rd in range(self.repeat):
                    Q = QuantumNetwork(configuration=self.configurations[exper][index][rd])
                    result = Q.run(save_tomography=save_tomo, save_result=save_result)
                    tmp_list_fidelity.append(result['fidelity'])
                    tmp_list_time_used.append(result['Time used'])
                self.Fidelity[exper].append(tmp_list_fidelity)
                self.Time_used[exper].append(tmp_list_time_used)

        self.executed = True

    def validate_run(self):

        # TODO implement validate checking system before actual execution
        passed = True

        return passed

    def get_fidelity(self, read_from=None):

        if self.executed is False and read_from is None:
            print('The experiment is not yet execute, please execute experiment.run() or provide result files')
            return None

        if read_from is not None:
            Fidelity = {}
            for exper in self.experiments:
                Fidelity[exper] = []
                for index in range(len(self.parameters_set)):
                    tmp_list = []
                    for rd in range(self.repeat):
                        with open(f"{read_from}/Result_{self.message_log}_exp{exper}_p{self.parameters_set[index]}_r{rd}.pkl", "rb") as f:
                            exp = dill.load(f)
                        tmp_list.append(exp['fidelity'])
                    Fidelity[exper].append(tmp_list)
            
            return Fidelity
        
        return self.Fidelity

    def tomography(self, read_from=None):

        # TODO implement logical tomography

        if self.executed is False and read_from is None:
            print('The experiment is not yet execute, please execute experiment.run() or provide result files')
            return None

        if read_from is not None:
            Fidelity = {}
            for exper in tqdm(self.experiments, desc='Tomography for experiments executed: ',disable=self.progress_bar):
                Fidelity[exper] = []
                for index in tqdm(range(len(self.parameters_set)), desc='Tomography for parameters executed: ',disable=self.progress_bar, leave=False):
                    tmp_list = []
                    for rd in range(self.repeat):
                        tomo_data = pd.read_csv(f"{read_from}/StateTomography_{self.message_log}_exp{exper}_p{self.parameters_set[index]}_r{rd}.csv")
                        tmp_list.append(Tomography(tomo_data))
                    Fidelity[exper].append(tmp_list)
            return Fidelity 


    def getDataFrame(self, read_from=None):

        if read_from is not None:
            Fidelity = self.get_fidelity(read_from=read_from)
        else:
            Fidelity = self.get_fidelity()

        df = []
        for index in range(len(self.parameters_set)):
            for rd in range(self.repeat):
                tmp = {exper: Fidelity[exper][index][rd] for exper in self.experiments}
                tmp['Probability (%)'] = self.parameters_set[index]
                df.append(tmp)
        
        return pd.DataFrame(df)

    def results(self):

        results = {}
        for exper in self.experiments:
            results[exper] = {}
            for index in range(len(self.parameters_set)):
                results[exper][index] = {}
                for rd in range(self.repeat):
                    with open(f"{self.path}/Result_{self.message_log}_exp{exper}_p{self.parameters_set[index]}_r{rd}.pkl", "rb") as f:
                        exp = dill.load(f)
                    results[exper][index][rd] = exp

        return results

    def raw_tomo_results(self):

        results = {}
        for exper in self.experiments:
            results[exper] = {}
            for index in range(len(self.parameters_set)):
                results[exper][index] = {}
                for rd in range(self.repeat):
                    results[exper][index][rd] = pd.read_csv(f"{self.path}/StateTomography_{self.message_log}_exp{exper}_p{self.parameters_set[index]}_r{rd}.csv")[['qubit1', 'qubit2']].to_dict('records')

        return results

    def plot_Expected_waiting_time(self, ):

        if self.executed is False:
            results = self.results()
        else:
            results = {}
            for exper in self.Time_used:
                results[exper] = {}
                for index in range(len(self.BSA_prob)):
                    results[exper][index] = {}
                    for rd in range(self.repeat):
                        results[exper][index][rd] = {'Time used': self.Time_used[exper][index][rd]}

        wt = {}
        for exper in results:
            wt[exper] = {}
            Prob_List = []
            WT_list = []
            for index in range(len(self.BSA_prob)):
                wt[exper][index] = {}
                for rd in range(self.repeat):
                    wt[exper][index][rd] = results[exper][index][rd]['Time used']

                    Prob_List.append(self.BSA_prob[index])
                    WT_list.append(wt[exper][index][rd])
    
            df = pd.DataFrame({'prob': Prob_List, 'trials': WT_list})
            sns.lineplot(data=df, x='prob', y='trials', ci='sd' ,label=exper)

        return 

    def plot_fidelity_history(self, figsize=None, row_col=None, analytic_fidelity=None):

        if analytic_fidelity is None:
            analytic_fidelity = {
                exper: {
                    index: 0
                for index in range(len(self.parameters_set))}
            for exper in self.experiments}

        # Extract information from result
        results = self.results()
        fidelity_hist = {}
        for exper in results:
            fidelity_hist[exper] = {}
            for index in range(len(self.parameters_set)):
                fidelity_hist[exper][index] = []
                for meas in range(len(results[exper][index][0]['Fidelity History'])):
                    for rd in range(self.repeat):
                        value = abs(results[exper][index][rd]['Fidelity History'][meas] - analytic_fidelity[exper][index])*100
                        tmp = {'meas': meas, f'rd': float(value)}
                        fidelity_hist[exper][index].append(tmp)
        # return fidelity_hist
        # Plotting
        sns.set_theme()
        if row_col is None:
            row_col = (1, len(self.experiments))
        if figsize is None:
            figsize = (int(18*row_col[0]), int(5*row_col[1]))
        fig, axes = plt.subplots(row_col[0], row_col[1], figsize=figsize)
        for row in range(row_col[0]):
            for col in range(row_col[1]):
                for index in range(len(self.parameters_set)):
                    exper = self.experiments[int(row_col[0]*row + col)]
                    text = f'paramters: {index}'
                    if row_col[0] == 1:
                        if row_col[1] == 1:
                            sns.lineplot(ax=axes, x='meas', y='rd', data=pd.DataFrame(fidelity_hist[exper][index]), ci='sd', label=text)
                            axes.set_title(exper)
                        else:
                            sns.lineplot(ax=axes[col], x='meas', y='rd', data=pd.DataFrame(fidelity_hist[exper][index]), ci='sd', label=text)
                            axes[col].set_title(exper)
                    else:
                        sns.lineplot(ax=axes[row, col], x='meas', y='rd', data=pd.DataFrame(fidelity_hist[exper][index]), ci='sd', label=text)
                        axes[row, col].set_title(exper)

        return fig, axes

    def plot(self, read_from=None):

        df = self.getDataFrame(read_from=read_from)

        sns.set_theme()
        for exper in self.experiments:
            fig = sns.lineplot(x='Probability (%)',y=exper, data=df, ci='sd', label=exper, markers=True, dashes=False)
        plt.ylabel('Fidelity')
        return fig

    def read_timeline_from_csv(file_path, excel=False, sheet_name=0):
        if excel == True:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            df = pd.read_csv(file_path)
        df['Edges'] = df['Edges'].transform(lambda x: ast.literal_eval(x))
        df['Num Trials'] = df['Num Trials'].transform(lambda x: True if x == 'True' else x)
        df['Num Trials'] = df['Num Trials'].transform(lambda x: int(x) if int(x) > 1 else x)
        return df.to_dict('records') ,df

    def str_to_list(x):
        return ast.literal_eval(x)


class Configuration:
    def __init__(self, topology, timeline, experiment=False, nodes_info=None, memFunc=None, gate_error=None, measurementError=None,
                 message=None, throughtput_edges=None, label_record='Physical', num_measured=9000, sim_time=None, 
                 collectFidelityHistory=False, result_path='result'):

        self.numPhysicalBuffer = 20
        self.numInternalEncodingBuffer = 20
        self.numInternalDetectingBuffer = 10
        self.numInternalInterfaceBuffer = 2
        self.memFunc = [1, 0, 0, 0] if memFunc is None else memFunc # Function of memory of qubit
        self.gateError = [1, 0, 0, 0] if gate_error is None else gate_error
        self.measurementError = 0 if measurementError is None else measurementError
        self.timeline = timeline
        self.experiment = experiment
        self.light_speed_in_fiber = 208189.206944 # km/s
        self.message = message
        self.num_measured = num_measured
        self.g = topology
        self.result_path = result_path
        self.label_recorded = label_record
        self.collectFidelityHistory = collectFidelityHistory
        self.simulation_time = sim_time
        self.coor_system = 'normal'
        
        # Initialize graph
        G = nx.Graph()
        for edge in topology:
            #time = topology[edge]['distance']/self.light_speed_in_fiber 
            G.add_edge(edge[0], edge[1]) # , weight=time

        nx.set_edge_attributes(G, topology)

        # Include function of error model
        if nodes_info is not None:
            self.nodes_info = nodes_info
            self.numPhysicalBuffer = self.nodes_info['numPhysicalBuffer']
            self.numInternalEncodingBuffer = self.nodes_info['numInternalEncodingBuffer']
            self.numInternalDetectingBuffer = self.nodes_info['numInternalDetectingBuffer']
            self.numInternalInterfaceBuffer = self.nodes_info['numInternalInterfaceBuffer']

        self.NetworkTopology = G
        if throughtput_edges is None:
            self.throughtputEdges = [list(G.nodes())[0], list(G.nodes())[-1]]
        else:
            self.throughtputEdges = throughtput_edges


class QuantumNetwork(_GeneratePhyscialResource.Mixin, 
                     _EntanglementPurification.Mixin, 
                     _EntanglementSwapping.Mixin, 
                     _GenerateLogicalResource.Mixin, 
                     _VirtualStateTomography.Mixin,
                     _TimeLag.Mixin):

    def __init__(self, configuration):

        self.configuration = configuration
        self.env = simpy.Environment()

        # Initialize QNICs in each nodes
        self.graph = self.configuration.NetworkTopology
        self.complete_graph = nx.complete_graph(self.graph.nodes)
        self.edges_list = [f'{node1}-{node2}' for node1, node2 in list(self.graph.edges())]
        self.complete_edges_list = [f'{node1}-{node2}' for node1, node2 in list(self.complete_graph.edges())]
        self.QuantumChannel = {edge: simpy.Store(self.env) for edge in self.edges_list}
        self.ClassicalChannel = {edge: simpy.Store(self.env) for edge in self.edges_list}

        self.table_name = {
            'externalQubitsTable': self.configuration.numPhysicalBuffer,
            'externalBusyQubitsTable': 0,
            'internalEncodingQubitTable': self.configuration.numInternalEncodingBuffer,
            'internalDetectingQubitTable': self.configuration.numInternalDetectingBuffer,
            'internalInterfaceQubitTable': self.configuration.numInternalInterfaceBuffer
        }

        self.QubitsTables = {
             table : { f'{node1}-{node2}': {
                f'QNICs-{node1}' : simpy.FilterStore(self.env),
                f'QNICs-{node2}' : simpy.FilterStore(self.env)
            } for node1, node2 in list(self.graph.edges()) }
        for table in self.table_name}

        for table in self.table_name:
            for node1, node2 in list(self.graph.edges()): # self.complete_graph.edges() #  
                for i in range(self.table_name[table]):
                    self.QubitsTables[table][f'{node1}-{node2}'] \
                    [f'QNICs-{node1}'].put(PhysicalQubit(node1, i, f'{node1}-{node2}', table[:8], self.env, table, self.configuration.memFunc, self.configuration.gateError, self.configuration.measurementError))
                    self.QubitsTables[table][f'{node1}-{node2}'] \
                    [f'QNICs-{node2}'].put(PhysicalQubit(node2, i, f'{node1}-{node2}', table[:8], self.env, table, self.configuration.memFunc, self.configuration.gateError, self.configuration.measurementError))

        self.internalLogicalQubitTable = { f'{node1}-{node2}': {
            f'QNICs-{node1}' : [],
            f'QNICs-{node2}' : []
        } for node1, node2 in list(self.graph.edges()) }

        self.resource_table_name = [
            'physicalResourceTable', 
            'internalPhysicalResourceTable', 
            'internalPurifiedResourceTable',
            'internalSecondPurifiedResourceTable',
            'logicalResourceTable'
        ]

        self.resourceTables = {}
        for table in self.resource_table_name:
            self.resourceTables[table] = {}
            for node1, node2 in list(self.complete_graph.edges()):
                if table[:8] == 'internal':
                    self.resourceTables[table][f'{node1}-{node2}'] = {
                        f'{node1}' : simpy.FilterStore(self.env),
                        f'{node2}' : simpy.FilterStore(self.env),
                    }
                else:
                    self.resourceTables[table][f'{node1}-{node2}'] = simpy.FilterStore(self.env) 

        #self.complete_edges_list = list(set(self.complete_edges_list))
        #self.edges_list = list(set(self.edges_list))
        for process in self.configuration.timeline:
            process['isSuccess'] = 0

        self.simulationLog = [] # <= TODO use this to collect data to plot
        self.qubitsLog = []
        self.numResrouceProduced = 0
        self.numBaseBellAttempt = 0
        self.connectionSetupTimeStamp = 0
        
        # For fidelity calculation
        self.measurementResult = []
        self.fidelityHistory = []
        self.Expectation_value = {
            'XX': {'commute': 0, 'anti-commute': 0},
            'YY': {'commute': 0, 'anti-commute': 0},
            'ZZ': {'commute': 0, 'anti-commute': 0}
        }
        
        self.fidelityStabilizerMeasurement = None

    '''
    Ulits
    '''

    def updateLog(self, log_message):
        self.simulationLog.append(log_message)
        return None

    def createLinkResource(self, node1, node2, resource1, resource2, resource_table, label='Physical', target_node=None, initial=False):
        resource1.isBusy, resource2.isBusy = True, True
        resource1.partner, resource2.partner = resource2, resource1
        resource1.partnerID, resource2.partnerID = resource2.qubitID, resource1.qubitID
        '''
        if initial == True:
            if resource1.initiateTime is not None or resource2.initiateTime is not None:
                raise ValueError('This qubit has not set free properly')
            resource1.initiateTime, resource2.initiateTime = self.env.now, self.env.now
        '''
        if label == 'Internal':
            resource1.initiateTime, resource2.initiateTime = self.env.now, self.env.now
            resource_table[f'{node1}-{node2}'][f'{target_node}'].put((resource1, resource2, 'Internal'))
            return None
        
        resource_table[f'{node1}-{node2}'].put((resource1, resource2, label))
        # self.updateLog({'Time': self.env.now, 'Message': f'Qubit ({resource1.qubitID}) entangle with Qubit ({resource2.qubitID})'})

        #if label == 'Physical': #self.configuration.label_recorded:
        #    self.numBaseBellAttempt += 1
        
        if (node1 and node2 in self.configuration.throughtputEdges) and label == self.configuration.label_recorded:
            #print(f'Resource created at {node1}-{node2}')
            # Record
            self.numResrouceProduced += 1

        # self.updateLog({'Time': self.env.now, 'Message': f'Qubit ({resource1.qubitID}) entangle with Qubit ({resource2.qubitID})'})

        return None

    def validateNodeOrder(self, node1, node2):

        if f'{node1}-{node2}' not in self.complete_edges_list:
            node1, node2 = node2, node1

        return node1, node2


    def Timeline(self):

        connectionSetup = [self.env.process(self.ConnectionSetup(self.configuration.throughtputEdges[0], 
                                                                 self.configuration.throughtputEdges[1]))]
        yield simpy.AllOf(self.env, connectionSetup)

        Unlimited_process = []
        Limited_process = []
        for process in self.configuration.timeline:
            if process['Main Process'] == '_GeneratePhysicalResource':
                p = [self.env.process(self.generatePhysicalResource(process['Edges'][0], process['Edges'][1], 
                                                                   label_out=process['Label out'],
                                                                   num_required=process['Num Trials']))]
            elif process['Main Process'] == '_EntanglementSwapping':
                p = [self.env.process(self.ExternalEntanglementSwapping( (process['Edges'][0], process['Edges'][1]), (process['Edges'][1], process['Edges'][2]), \
                                                                         num_required=process['Num Trials'], \
                                                                         label_in=process['Label in'], \
                                                                         label_out=process['Label out'], \
                                                                         resource_type=process['Resource Type']))]
            elif process['Main Process'] == '_StateTomography':
                p = [self.env.process(self.VirtualStateTomography(process['Edges'][0], process['Edges'][1], \
                                                                 num_required=process['Num Trials'], \
                                                                 label_in=process['Label in'], \
                                                                 resource_type=process['Resource Type']))]
            elif process['Main Process'] == '_Purification':
                p = [self.env.process(self.Purification(process['Edges'][0], process['Edges'][1], 
                                                       num_required=process['Num Trials'], \
                                                       label_in=process['Label in'], \
                                                       label_out=process['Label out'], \
                                                       protocol=process['Protocol']))]
            elif process['Main Process'] == '_GenerateLogicalResource':
                p = [self.env.process(self.generateLogicalResource(process['Edges'][0], process['Edges'][1], 
                                                                  num_required=process['Num Trials'], \
                                                                  label_in=process['Label in'], \
                                                                  label_out=process['Label out'], \
                                                                  protocol=process['Protocol']))]
            elif process['Main Process'] == '_GeneratePhysicalResourceEPPS':
                p = [self.env.process(self.generatePhysicalResourceEPPS(process['Edges'][0], process['Edges'][2], process['Edges'][1],
                                                                               label_out=process['Label out'],
                                                                               num_required=process['Num Trials']))]
            elif process['Main Process'] == '_GeneratePhysicalResourceEPPSPulse':
                p = [self.env.process(self.generatePhysicalResourceEPPSPulse(process['Edges'][0], process['Edges'][2], process['Edges'][1],
                                                                                    label_out=process['Label out'],
                                                                                    num_required=process['Num Trials']))]
            elif process['Main Process'] == '_GeneratePhysicalResourcePulse':
                p = [self.env.process(self.generatePhysicalResourceStaticPulse(process['Edges'][0], process['Edges'][1], 
                                                                            label_out=process['Label out'],
                                                                            num_required=process['Num Trials']))]
            elif process['Main Process'] in ['PrototypeGeneratePhysicalResourcePulse', 'Generate physical Bell pair']: 
                p = [
                    self.env.process(self.Emitter(process['Edges'][0], process['Edges'][1], 
                                                  label_out=process['Label out'],
                                                  num_required=process['Num Trials'])),
                    self.env.process(self.Detector(process['Edges'][0], process['Edges'][1], 
                                                  label_out=process['Label out'],
                                                  num_required=process['Num Trials'])),
                    self.env.process(self.ClassicalMessageHandler(process['Edges'][0], process['Edges'][1], 
                                                  label_out=process['Label out'],
                                                  num_required=process['Num Trials']))
                ]
            elif process['Main Process'] in ['PrototypeGeneratePhysicalResourcePulseMidpoint', 'Generate physical Bell pair (Midpoint)']: 
                p = [
                    self.env.process(self.Emitter(process['Edges'][0], process['Edges'][2], 
                                                  label_out=process['Label out'],
                                                  num_required=process['Num Trials'], middleNode=process['Edges'][1], EPPS=process['Protocol'])),
                    self.env.process(self.Detector(process['Edges'][0], process['Edges'][2], 
                                                  label_out=process['Label out'],
                                                  num_required=process['Num Trials'], middleNode=process['Edges'][1], EPPS=process['Protocol'])),
                    self.env.process(self.ClassicalMessageHandler(process['Edges'][0], process['Edges'][2], 
                                                  label_out=process['Label out'],
                                                  num_required=process['Num Trials'], middleNode=process['Edges'][1]))
                ]
            elif process['Main Process'] in ['PrototypeEntanglementSwapping', 'Entanglement swapping']:
                p = [self.env.process(self.PrototypeExternalEntanglementSwapping(process,( process['Edges'][0], process['Edges'][1]), (process['Edges'][1], process['Edges'][2]), \
                                                                                num_required=process['Num Trials'], \
                                                                                label_in=process['Label in'], \
                                                                                label_out=process['Label out'], \
                                                                                resource_type=process['Resource Type']))]    
            elif process['Main Process'] in ['PrototypeStateTomography', 'State tomography']:
                p = [self.env.process(self.PrototypeVirtualStateTomography(process, process['Edges'][0], process['Edges'][1], \
                                                                        num_required=process['Num Trials'], \
                                                                        label_in=process['Label in'], \
                                                                        resource_type=process['Resource Type']))] 
            elif process['Main Process'] in ['PrototypePurification', 'Entanglement purification']:
                p = [self.env.process(self.PrototypePurification(process, process['Edges'][0], process['Edges'][1], 
                                                       num_required=process['Num Trials'], \
                                                       label_in=process['Label in'], \
                                                       label_out=process['Label out'], \
                                                       protocol=process['Protocol']))]   
            elif process['Main Process'] in ['PrototypeGenerateLogicalResource', 'Generate logical Bell pair']:
                p = [self.env.process(self.PrototypeGenerateLogicalResource(process, process['Edges'][0], process['Edges'][1], 
                                                                  num_required=process['Num Trials'], \
                                                                  label_in=process['Label in'], \
                                                                  label_out=process['Label out'], \
                                                                  protocol=process['Protocol']))]                                                                                              
            else:
                raise ValueError('Process is not define.')
            if type(process['Num Trials']) is int:
                Limited_process += p
            else:
                Unlimited_process += p

        yield simpy.AllOf(self.env, Limited_process)

        #connectionSetup = [self.env.process(self.ConnectionSetup(self.configuration.throughtputEdges[0], 
        #                                                         self.configuration.throughtputEdges[1]))]
        #yield simpy.AllOf(self.env, connectionSetup)

        self.connectionSetupTimeStamp = self.env.now - self.connectionSetupTimeStamp

    def run(self, save_tomography=False, save_result=True):

        timeline = self.env.process(self.Timeline())
        sim_time = timeline if self.configuration.simulation_time is None else self.configuration.simulation_time
        self.env.run(until=sim_time)

        # Store simulated data 

        # Create folder if not exist
        if save_tomography is True or save_result is True:
            if (not os.path.exists(f'{self.configuration.result_path}')):
                os.makedirs(f'{self.configuration.result_path}')
        
        if save_tomography is True:
            # Data for state-tomography
            pd.DataFrame(self.measurementResult).to_csv(f'{self.configuration.result_path}/StateTomography_{self.configuration.experiment}_{self.configuration.message}.csv')


        # Data for configuration and fidelity using stabilizer couting method
        
        #result_file = open(f"{self.configuration.result_path}/Result_{self.configuration.experiment}_{self.configuration.message}.json", "w")
        #config = {data: self.configuration.__dict__[data] for data in self.configuration.__dict__ if data not in ['NetworkTopology'] }
        #json.dump(config, result_file)
        #result_file.close()

        config = {data: self.configuration.__dict__[data] for data in self.configuration.__dict__}
        config['fidelity'] = self.fidelityStabilizerMeasurement
        config['Resources Produced'] = self.numResrouceProduced
        config['Base Resources Produced'] = self.numBaseBellAttempt
        config['Time used'] = self.connectionSetupTimeStamp
        config['Fidelity History'] = self.fidelityHistory
        config['Qubits waiting time'] = self.qubitsLog

        # Save log data
        config['Simulation log'] = self.simulationLog

        if save_result is True:
            with open(f"{self.configuration.result_path}/Result_{self.configuration.experiment}_{self.configuration.message}.pkl", "wb") as f:
                dill.dump(config, f)

        return config
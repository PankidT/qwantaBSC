from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
import pandas as pd 
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from copy import deepcopy
import qiskit.quantum_info as qi
import ast
import numpy as np

def encoderInitiater(List=[1, 2, 3, 4, 5, 6, 0]):
    a, b, c, d, e, f, i = List
    # Defined composite encoding gate of Steane code
    qr_a = QuantumRegister(7, name='a')
    qc = QuantumCircuit(qr_a, name='encoder')
    ### Encode
    qc.h([qr_a[a], qr_a[b], qr_a[c]])
    qc.cx(qr_a[i], qr_a[d]); qc.cx(qr_a[i], qr_a[e]);qc.barrier();
    qc.cx(qr_a[c], qr_a[i]);qc.cx(qr_a[c], qr_a[d]);qc.cx(qr_a[c], qr_a[f]);qc.barrier();
    qc.cx(qr_a[b], qr_a[i]);qc.cx(qr_a[b], qr_a[e]);qc.cx(qr_a[b], qr_a[f]);qc.barrier();
    qc.cx(qr_a[a], qr_a[d]);qc.cx(qr_a[a], qr_a[e]);qc.cx(qr_a[a], qr_a[f]);qc.barrier();
    qc.draw(output='mpl', fold=50, style={'showindex':True})#.savefig('SteaneEncoder.png')
    return qc.to_instruction()

encoder = encoderInitiater([3, 1, 0, 4, 5, 6, 2])

H = np.array([[0, 1, 1, 0, 0, 1, 1],
              [0, 0, 0, 1, 1, 1, 1],
              [1, 0, 1, 0, 1, 0, 1]])

'''
    Convert string of raw result from measurement to vector
    Args
        raw (str) - String of raw result from measurement on encoded qubits
        reverse (bool) - reverse the input raw or not? default = True 
    return binary vector 
'''
def RawtoVec(raw, reverse=False):
    vec = np.zeros((7, 1))
    if reverse == True:
        raw = raw[::-1]
    for i, j in enumerate(raw):
        vec[i][0] = int(j)
    return vec


'''
    Compute Hv
    Args
        vector (numpy array) - vector of raw logical readout
        ParityMatrix (numpy array) - A parity check matrix 
    return Hv mod 2 (numpy array)
'''
def ParityCheck(vector, ParityMatrix):
    return ParityMatrix.dot(vector) % 2

'''
    Return index of bit flip error
    Args
        vector (numpy array) - result from ParityCheck
    return index of bit flip error (int)
'''
def ErrorPointer(vec):
    return int(''.join([str(int(i[0])) for i in vec.tolist()]), 2)

def Decode(raw_result, H):
    vec = RawtoVec(raw_result, True)
    inter = ParityCheck(vec, H)
    pointer = ErrorPointer(inter)
    if raw_result.count('1') % 2 == 1:
        result = True
    else:
        result = False
    if pointer == 0:
        pass
        #print('Raw result: ', raw_result[::-1], ' with no bit flip error')
        #print('The logical readout is: ', int(result))
    else:
        result = not result
        #print('Raw result: ', raw_result[::-1], ' with bit flip error occur at: ', pointer)
        #print('The logical readout is: ', int(result))
    return result

def LogicalTomography(data, measurement_error=0, basis_measure_num=1000):

    measurement_basis = ['XX', 'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ'] 
    x = np.random.random((int(len(measurement_basis*basis_measure_num)), 7))
    y = np.random.random((int(len(measurement_basis*basis_measure_num)), 7))
    all_results = []
    for index, basis in enumerate(measurement_basis):
        measurement_outcome = {}
        for operator in range(basis_measure_num):
            operator_index = index*basis_measure_num + operator
            
            qr_1 = QuantumRegister(7)
            cr_1 = ClassicalRegister(7)
            qr_2 = QuantumRegister(7)
            cr_2 = ClassicalRegister(7)
            qc = QuantumCircuit(qr_1, qr_2, cr_1, cr_2)

            qc.append(encoder, qr_1)
            qc.append(encoder, qr_2)

            qc.h(qr_1)

            qc.cx(qr_1, qr_2)

            for i, j in enumerate(ast.literal_eval(data['qubit1'][operator_index])):
                if j == 1:
                    qc.x(qr_1[i])
                    qc.z(qr_1[i])
                elif j == 2:
                    qc.x(qr_1[i])
                elif j == 3:
                    qc.z(qr_1[i])
            for i, j in enumerate(ast.literal_eval(data['qubit2'][operator_index])):
                if j == 1:
                    qc.x(qr_2[i])
                    qc.z(qr_2[i])
                elif j == 2:
                    qc.x(qr_2[i])
                elif j == 3:
                    qc.z(qr_2[i])
            qc.barrier()
            # measurement basis 
            if basis[0] == 'X':
                qc.h(qr_1)
                # apply measurement error
                for qubit in range(7):
                    if x[operator_index][qubit] < measurement_error:
                        qc.x(qr_1[qubit])
            elif basis[0] == 'Y':
                qc.sdg(qr_1) # sdg
                qc.h(qr_1)
                # apply measurement error
                for qubit in range(7):
                    if x[operator_index][qubit] < measurement_error:
                        qc.x(qr_1[qubit])
            elif basis[0] == 'Z':
                # apply measurement error
                for qubit in range(7):
                    if x[operator_index][qubit] < measurement_error:
                        qc.x(qr_1[qubit])

            if basis[1] == 'X':
                qc.h(qr_2)
                # apply measurement error
                for qubit in range(7):
                    if x[operator_index][qubit] < measurement_error:
                        qc.x(qr_2[qubit])
            elif basis[1] == 'Y':
                qc.sdg(qr_2)
                qc.h(qr_2)
                # apply measurement error
                for qubit in range(7):
                    if x[operator_index][qubit] < measurement_error:
                        qc.x(qr_2[qubit])
            elif basis[1] == 'Z':
                # apply measurement error
                for qubit in range(7):
                    if x[operator_index][qubit] < measurement_error:
                        qc.x(qr_2[qubit])

            qc.measure(qr_1, cr_1)
            qc.measure(qr_2, cr_2)
            raw_result = execute(qc, Aer.get_backend('qasm_simulator'), shots=1).result().get_counts()
            
            s = ''
            for i in list(raw_result.keys())[0].split(' '):
                s += str(int(Decode(i, H)))
            decoded_result = {s: 1}

            for res in decoded_result:
                if res in measurement_outcome:
                    measurement_outcome[res] += 1
                else:
                    measurement_outcome[res] = 1
        all_results.append(measurement_outcome)

    # Bell pair circuit

    qr_bell = QuantumRegister(2)
    qc_bell = QuantumCircuit(qr_bell)
    qc_bell.h(qr_bell[0])
    qc_bell.cx(qr_bell[0], qr_bell[1])
    qc_bell.barrier()

    target_state_bell = qi.Statevector.from_instruction(qc_bell)

    qst_qc = state_tomography_circuits(qc_bell, [qr_bell[0],qr_bell[1]])
    #Run in Aer
    job = execute(qst_qc, Aer.get_backend('qasm_simulator'), shots=basis_measure_num)
    raw_results = job.result()

    new_result = deepcopy(raw_results)

    for resultidx, _ in enumerate(raw_results.results):
        new_result.results[resultidx].data.counts = all_results[resultidx]

    tomo_bell = StateTomographyFitter(new_result, qst_qc)
    # Perform the tomography fit
    # which outputs a density matrix
    rho_fit_bell = tomo_bell.fit(method='lstsq')

    F_bell = qi.state_fidelity(rho_fit_bell, target_state_bell)
    return F_bell
import copy
import os
import re
import time
from datetime import datetime

import numpy as np
from quafu import QuantumCircuit
# from quafu import simulate
from quafu.algorithms.gradient import grad_adjoint, grad_para_shift, grad_finit_diff
from quafu.algorithms.hamiltonian import Hamiltonian
from quafu.algorithms.optimizer import adam
from quafu.simulators.simulator import SVSimulator
from scipy.optimize import minimize

from qusteed.applications.utils.gradient_calculator import grad_para_shift_qpu
from qusteed.applications.utils.gradient_calculator import spsa
from qusteed.applications.utils.load_module import load_module
from qusteed.applications.utils.public import group_hamiltonian_terms, calculate_expectation
from qusteed.applications.utils.update_paras import order_variables_indices, update_circuit_paras
from qusteed.results.vqa_results import VQAResult, save_results, load_results
from qusteed.security_check.check_user_code import bandit_check, high_risk_check


# TODO: Unified VQA base class
class VQE:
    def __init__(self,
                 ansatz_circuit: QuantumCircuit | str,
                 original_circuit: QuantumCircuit | str,
                 hamiltonian: Hamiltonian,
                 initial_parameters: list = None,
                 maxiter: int = 150,
                 default_gradient: str = "reverse-mode",
                 default_backend: str = "simulator",
                 user_gradient=None,
                 user_backend=None,
                 backend=None,
                 mapping_req2q: dict = None,
                 initial_variables: list = None,
                 initial_v2q: dict = None,
                 final_v2q: dict = None,
                 final_q2c: dict = None,
                 correction_matrices=None,
                 minimizer="spsa",
                 token: str = None,
                 repeat: int = None,
                 taskid=None
                 ):
        """
        Args:
            default_gradient (str): 'simulator-para-shift', 'qpu-para-shift', 'finite-diff', 'reverse-mode'
            default_backend (str): 'simulator', 'local_qpu'
            user_gradient:
            user_backend: The backend interface function used for quantum circuit sampling,
                          which supports at least the input parameter qc and shots, and "qc" types include:
                          quafu.QuantumCircuit | qiskit.QuantumCircuit ｜ OpenQASM 2.0.
                          The return result must be of dict type, res={'00': 2,'01': 3,'10': 4,'11': 1}.
                          For example, res=backend(qc, shots=3000),
            minimizer (str): "spsa", "adam", "scipy"

        Returns:

        """
        self.ansatz_circuit = None
        self.original_circuit = None
        self.hamiltonian = hamiltonian
        self.initial_parameters = initial_parameters
        self.maxiter = maxiter
        self.default_gradient = default_gradient
        self.default_backend = default_backend
        self.user_gradient = user_gradient
        self.user_backend = user_backend
        self._bounds = None
        self._iter_num = 0
        self._spsa_iter_num = 0
        self._cost_values = []
        self._optimization_parameters = []
        self._optimization_time = 0
        # self._error = None
        self._vqe_results = VQAResult()
        self._start_time = 0
        self.cost_fun = None
        self.gradient_fun = None
        # self._filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._taskid = taskid
        if taskid is None:
            self._filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        else:
            self._filename = str(taskid)
        self._backend = backend
        self._mapping_req2q = mapping_req2q
        self._initial_v2q = initial_v2q
        self._final_v2q = final_v2q
        self._final_q2c = final_q2c
        self._get_ansatz_circuit(ansatz_circuit)
        self._get_original_circuit(original_circuit)
        self.measure_circuits = None
        self.real_measure_circuit(self.hamiltonian)
        self.initial_variables = initial_variables
        self._order_indices = None
        self._minimizer = minimizer
        self._correction_matrices = correction_matrices
        self._token = token
        self._repeat = repeat
        # self._shots = shots

    def _get_ansatz_circuit(self, ansatz_circuit):
        if isinstance(ansatz_circuit, QuantumCircuit):
            self.ansatz_circuit = ansatz_circuit
        elif isinstance(ansatz_circuit, str):
            qubit_num = int(re.findall(r"\d+\.?\d*", ansatz_circuit.split('qreg')[1].split(';')[0])[0])
            self.ansatz_circuit = QuantumCircuit(qubit_num)
            self.ansatz_circuit.from_openqasm(ansatz_circuit)
        else:
            raise TypeError("ansatz_circuit is not quafu circuit or openqasm 2.0")

    def _get_original_circuit(self, original_circuit):
        if isinstance(original_circuit, QuantumCircuit):
            self.original_circuit = original_circuit
        elif isinstance(original_circuit, str):
            qubit_num = int(re.findall(r"\d+\.?\d*", original_circuit.split('qreg')[1].split(';')[0])[0])
            self.original_circuit = QuantumCircuit(qubit_num)
            self.original_circuit.from_openqasm(original_circuit)
        else:
            raise TypeError("original_circuit is not quafu circuit or openqasm 2.0")

    def real_measure_circuit(self, hamiltonian: Hamiltonian):
        """
        Args:
            hamiltonian (Hamiltonian): measure base and its positions.
        """
        # TODO: Classify Hamiltonian terms based on their measurement strategies.

        if isinstance(self._mapping_req2q, dict) and len(self._mapping_req2q) > 0:
            qubits = [q for req, q in self._mapping_req2q.items()]
        else:
            qubits = [q for q in range(self.ansatz_circuit.num)]
            # assert ValueError(str(self._mapping_req2q) + "is not a dict.")

        self._final_q2v = {q: v for v, q in self._final_v2q.items()}
        self._v2q = {self._final_q2v[req]: q for req, q in self._mapping_req2q.items()}

        self._initial_q2v = {q: v for v, q in self._initial_v2q.items()}
        self._inv2q = {self._initial_q2v[req]: q for req, q in self._mapping_req2q.items()}

        self._hamv2q = {v: self._final_q2c[self._v2q[v]] for v, q in self._v2q.items()}

        self._end_v2c = {}
        self._mapping_q2req = {q: req for req, q in self._mapping_req2q.items()}
        # print('self._mapping_q2req', self._mapping_q2req)
        for v, q in self._inv2q.items():
            c = self._final_q2c[q]
            self._end_v2c[self._mapping_q2req[self._v2q[v]]] = c
        # print('self._end_v2c', self._end_v2c)

        measure_circuits = []
        groups = group_hamiltonian_terms(hamiltonian)
        for basis, terms in groups.items():
            measure_circuit = QuantumCircuit(self.ansatz_circuit.num)

            if basis == "X":
                [measure_circuit.ry(q, -np.pi / 2) for q in qubits]
                measure_circuits.append((measure_circuit, terms))
            elif basis == "Y":
                [measure_circuit.rx(q, np.pi / 2) for q in qubits]
                measure_circuits.append((measure_circuit, terms))
            elif basis == "Z":
                measure_circuits.append((measure_circuit, terms))
            elif basis == "Mixed":
                for term in terms:
                    measure_circuit = QuantumCircuit(self.ansatz_circuit.num)
                    for i in range(len(term.pos)):
                        if term.paulistr[i] == "X":
                            measure_circuit.ry(self._v2q[term.pos[i]], -np.pi / 2)
                        elif term.paulistr[i] == "Y":
                            measure_circuit.rx(self._v2q[term.pos[i]], np.pi / 2)
                    measure_circuit.draw_circuit()
                    measure_circuits.append((measure_circuit, [term]))
        self.measure_circuits = measure_circuits
        return measure_circuits

    # # Define a callback function to record the optimization process
    # def callback(self, x):
    #     global history
    #     global iter_count
    #     fval = self.cost_qpu(x)
    #     history['fun'].append(fval)
    #     iter_count += 1
    #     print(f"Iteration {iter_count}, Energy: {fval}")

    def run(self) -> VQAResult:
        """
        """
        try:
            self._start_time = time.time()
            self._order_indices = order_variables_indices(self.ansatz_circuit, self.initial_variables)
            self._vqe_results.initial_parameters = self.initial_parameters
            if self.default_backend == "simulator" and self.user_backend is None:
                self.cost_fun = self.cost_simulator
            elif self.user_backend is not None:
                if callable(self.user_backend):
                    self.cost_fun = self.cost_general_backend
                else:
                    raise TypeError(self.user_backend.__name__ + "cannot be called, may not be a function.")
            else:
                raise TypeError("No available backend given.")

            if self.default_gradient == "reverse-mode" and self.user_gradient is None:
                self.gradient_fun = self.grad_reverse
            elif self.default_gradient == "finite-diff" and self.user_gradient is None:
                self.gradient_fun = self.grad_diff
            elif self.default_gradient == "simulator-para-shift" and self.user_gradient is None:
                self.gradient_fun = self.grad_para_simulator
            elif self.default_gradient == "qpu-para-shift" and self.user_gradient is None:
                self.gradient_fun = self.grad_para_qpu
            else:
                raise TypeError("No available gradient evaluator given.")

            self.ansatz_circuit.get_parameter_grads()
            self.original_circuit.get_parameter_grads()
            if self.initial_parameters is None:
                self.initial_parameters = np.random.rand(len(self.ansatz_circuit.variables))

            self._bounds = [(-np.pi, np.pi) for _ in range(len(self.initial_parameters))]

            if self._minimizer == "scipy":
                minimize(self.cost_fun,
                         np.array(self.initial_parameters),
                         method='L-BFGS-B',
                         # method='newton-cg',
                         # method='COBYLA',
                         jac=self.gradient_fun,
                         bounds=self._bounds,
                         options={'maxiter': self.maxiter, 'ftol': 1e-10, 'gtol': 1e-10})
            elif self._minimizer == "adam":
                minimize(self.cost_fun,
                         np.array(self.initial_parameters),
                         method='L-BFGS-B',
                         jac=self.gradient_fun,
                         bounds=self._bounds,
                         options={'maxiter': 10})
                adam(self.cost_fun, self._optimization_parameters[-1], self.gradient_fun, verbose=False,
                     maxiter=self.maxiter - self._iter_num - 1)
            elif self._minimizer == "spsa":
                spsa(self.cost_fun, np.array(self.initial_parameters), learning_rate=0.1, perturbation=0.05,
                     num_iterations=self.maxiter)
            elif callable(self._minimizer):
                self._minimizer(self.cost_fun, np.array(self.initial_parameters), self.maxiter)
            elif isinstance(self._minimizer, str) and self._minimizer not in ["scipy", "adam", "spsa"]:
                # Check for any high risk system operations
                if high_risk_check(self._minimizer):
                    module_file = "user_module_" + str(self._taskid) + ".py"
                    user_module_file = os.path.join(os.getcwd(), "user_module", module_file)
                    os.makedirs(os.path.dirname(user_module_file), exist_ok=True)
                    with open(user_module_file, 'w') as file:
                        file.write(self._minimizer)
                    # Check for any security issues
                    if bandit_check(user_module_file):
                        module = load_module(user_module_file, "user_module")
                        module.my_minimizer(self.cost_fun, np.array(self.initial_parameters), self.maxiter)
            else:
                raise Exception("minimizer is not define")

            return load_results(self._filename)
        except Exception as e:
            print(f"Exception occurred: {e}")
            self._vqe_results.error = e
            save_results(self._vqe_results, self._filename)
            return load_results(self._filename)

    def _update_results(self):
        self._vqe_results.initial_parameters = self.initial_parameters
        self._vqe_results.optimization_time = time.time() - self._start_time
        self._vqe_results.optimization_parameters = self._optimization_parameters
        self._vqe_results.cost_values = self._cost_values
        if self._minimizer == "spsa":
            self._vqe_results.max_iteration = self._spsa_iter_num
        else:
            self._vqe_results.max_iteration = self._iter_num
        if self._iter_num > 0:
            self._vqe_results.optimal_value = self._cost_values[-1]
            self._vqe_results.optimal_parameters = self._optimization_parameters[-1]
        print(f"Iteration {self._iter_num}, Energy: {self._cost_values[-1]}")

    def cost_simulator(self, x):
        self._iter_num += 1
        self._optimization_parameters.append(x)
        self.ansatz_circuit._update_params(x)
        eigenvalue = sum(SVSimulator().run(self.ansatz_circuit, hamiltonian=self.hamiltonian)["pauli_expects"])
        self._cost_values.append(eigenvalue)
        self._update_results()
        save_results(self._vqe_results, self._filename)
        return eigenvalue

    def cost_general_backend(self, x):
        eigenvalue = 0
        self._iter_num += 1
        for qc, terms in self.measure_circuits:
            nqc = update_circuit_paras(self.ansatz_circuit, x, self._order_indices)
            measure_qc = self._add_measure(nqc, qc)
            res = self.user_backend(measure_qc.to_openqasm(),
                                    backend=self._backend,
                                    correction_matrices=self._correction_matrices,
                                    token=self._token,
                                    repeat=self._repeat)
            for term in terms:
                pos = [self._hamv2q[p] for p in term.pos]
                eigenvalue += term.coeff * calculate_expectation(sampling_results=res, positions=pos)
        if self._minimizer == "spsa":
            if self._iter_num % 3 == 1:
                self._spsa_iter_num += 1
                self._optimization_parameters.append(x)
                self._cost_values.append(eigenvalue)
                self._update_results()
        else:
            self._optimization_parameters.append(x)
            self._cost_values.append(eigenvalue)
            self._update_results()
        save_results(self._vqe_results, self._filename)
        return eigenvalue

    # TODO: Check qpu support, may need to change to physical qubits
    def grad_para_qpu(self, x):
        grads = []
        for qc, terms in self.measure_circuits:
            # qc = copy.deepcopy(self.ansatz_circuit)
            # qc._update_params(x)
            qc = update_circuit_paras(qc, x, self._order_indices)
            ham = Hamiltonian(terms)
            grads.append(grad_para_shift_qpu(qc, ham, backend=self.user_backend))
        grads = [sum(values) for values in zip(*grads)]
        return grads

    def grad_para_simulator(self, x):
        self.original_circuit._update_params(x)
        return grad_para_shift(self.original_circuit, self.hamiltonian)

    def grad_diff(self, x):
        self.original_circuit._update_params(x)
        return grad_finit_diff(self.original_circuit, self.hamiltonian)

    def grad_reverse(self, x):
        # self.ansatz_circuit._update_params(x)
        self.ansatz_circuit = update_circuit_paras(self.ansatz_circuit, x, self._order_indices)
        return grad_adjoint(self.ansatz_circuit, self.hamiltonian)

    def _add_measure(self, ansatz_circuit: QuantumCircuit, measure_circuit: QuantumCircuit):
        if len(measure_circuit.gates) > 0:
            gates = ansatz_circuit.gates
            copy_measure = copy.deepcopy(ansatz_circuit.measures)
            measure_qlist = [q for q in list(ansatz_circuit.measures.keys())]
            if gates[-1].name == "barrier":
                ansatz_circuit.gates.pop(-1)
                ansatz_circuit.measures = {}
                ansatz_circuit.add_gates(measure_circuit.gates)
                ansatz_circuit.barrier(measure_qlist)
                ansatz_circuit.measure(list(copy_measure.keys()), list(copy_measure.values()))
            else:
                ansatz_circuit.measures = {}
                ansatz_circuit.add_gates(measure_circuit.gates)
                ansatz_circuit.barrier(measure_qlist)
                ansatz_circuit.measure(list(copy_measure.keys()), list(copy_measure.values()))
        return ansatz_circuit
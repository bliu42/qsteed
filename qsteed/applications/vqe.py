import copy
import time

import numpy as np
from quafu import QuantumCircuit
# from quafu import simulate
from quafu.algorithms.gradient import grad_adjoint, grad_para_shift, grad_finit_diff
from quafu.algorithms.hamiltonian import Hamiltonian
from quafu.algorithms.optimizer import adam
from quafu.simulators.simulator import SVSimulator
from quark import connect
from scipy.optimize import minimize

from qusteed.applications.utils.public import group_hamiltonian_terms, calculate_expectation
from qusteed.results.vqa_results import VQAResult
from qusteed.applications.utils.gradient_calculator import grad_para_shift_qpu
# import spsa
from qusteed.applications.utils.gradient_calculator import spsa


# TODO: Unified VQA base class
# TODO: Define the return result class and output the result information
class VQE:
    def __init__(self, ansatz_circuit: QuantumCircuit, hamiltonian: Hamiltonian, initial_parameters: list = None,
                 maxiter: int = 300, shots: int = 3000,
                 default_gradient: str = "reverse-mode", default_backend: str = "simulator",
                 user_gradient=None, user_backend=None):
        """
        Args:
            default_gradient (str): 'simulator-para-shift', 'qpu-para-shift', 'finite-diff', 'reverse-mode'
            default_backend (str): 'simulator', 'local_qpu'
            user_gradient:
            user_backend: The backend interface function used for quantum circuit sampling,
                          which supports at least the input parameter qc and shots, and "qc" types include:
                          quafu.QuantumCircuit | qiskit.QuantumCircuit ｜ OpenQASM 2.0.
                          The return result must be of dict type, res={'00': 2,'01': 3,'10': 4,'11': 1}.
                          For example, res=backend(qc, shots=3000)

        Returns:

        """
        self.ansatz_circuit = ansatz_circuit
        self.hamiltonian = hamiltonian
        self.initial_parameters = initial_parameters
        self.maxiter = maxiter
        self.default_gradient = default_gradient
        self.default_backend = default_backend
        self.user_gradient = user_gradient
        self.user_backend = user_backend
        self.measure_circuits = self.real_measure_circuit(self.ansatz_circuit, self.hamiltonian)
        self._bounds = None
        self._iter_num = 0
        self._cost_values = []
        self._optimization_parameters = []
        self._optimization_time = 0
        self.cost_fun = None
        self.gradient_fun = None

    def compile_ansatz_circuit(self):
        pass

    def real_measure_circuit(self, circuit, hamiltonian: Hamiltonian):
        """

        Args:
            circuit (QuantumCircuit): Quantum circuit that need to be executed on backend.
            hamiltonian (Hamiltonian): measure base and its positions.
        """
        # TODO: Classify Hamiltonian terms based on their measurement strategies.
        measure_circuits = []
        groups = group_hamiltonian_terms(hamiltonian)
        for basis, terms in groups.items():
            measure_circuit = copy.deepcopy(circuit)
            measure_circuit.measures = {}
            # TODO 真实的比特
            qubit = [0, 1, 2]
            if basis == "X":
                [measure_circuit.ry(qubit[q], -np.pi / 2) for q in qubit]
                measure_circuit.measure(qubit, qubit)
                measure_circuit.get_parameter_grads()
                measure_circuits.append((measure_circuit, terms))
            elif basis == "Y":
                [measure_circuit.rx(qubit[q], np.pi / 2) for q in qubit]
                measure_circuit.measure(qubit, qubit)
                measure_circuit.get_parameter_grads()
                measure_circuits.append((measure_circuit, terms))
            elif basis == "Z":
                measure_circuit.measure(qubit, qubit)
                measure_circuit.get_parameter_grads()
                measure_circuits.append((measure_circuit, terms))
            elif basis == "Mixed":
                # TODO: 逻辑比特到物理比特映射后，相应的terms里的比特位置也要变为物理比特。
                for term in terms:
                    for i in range(len(term.pos)):
                        if term.paulistr[i] == "X":
                            measure_circuit.ry(term.pos[i], -np.pi / 2)
                        elif term.paulistr[i] == "Y":
                            measure_circuit.rx(term.pos[i], np.pi / 2)
                    measure_circuit.measure(qubit, qubit)
                    measure_circuit.get_parameter_grads()
                    measure_circuits.append((measure_circuit, [term]))
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
        start_time = time.time()

        if self.default_backend == "simulator" and self.user_backend is None:
            self.cost_fun = self.cost_simulator
        elif self.default_backend == "local_qpu" and self.user_backend is None:
            self.cost_fun = self.cost_local_qpu
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
        if self.initial_parameters is None:
            self.initial_parameters = np.random.rand(len(self.ansatz_circuit.variables))

        self._bounds = [(-np.pi, np.pi) for _ in range(len(self.initial_parameters))]

        if self.maxiter <= 100:
            # minimize(self.cost_fun,
            #          np.array(self.initial_parameters),
            #          method='L-BFGS-B',
            #          # method='newton-cg',
            #          jac=self.gradient_fun,
            #          bounds=self._bounds,
            #          options={'maxiter': self.maxiter, 'ftol': 1e-15, 'gtol': 1e-15})
            spsa(self.cost_fun, np.array(self.initial_parameters), 0.1, 0.1, num_iterations=100)

        else:
            minimize(self.cost_fun,
                     np.array(self.initial_parameters),
                     method='L-BFGS-B',
                     jac=self.gradient_fun,
                     bounds=self._bounds,
                     options={'maxiter': 10})
            adam(self.cost_fun, self._optimization_parameters[-1], self.gradient_fun, verbose=False,
                 maxiter=self.maxiter - self._iter_num - 1)

        vqe_results = VQAResult()
        vqe_results.initial_parameters = self.initial_parameters
        vqe_results.optimization_time = time.time() - start_time
        vqe_results.optimization_parameters = self._optimization_parameters
        vqe_results.cost_values = self._cost_values
        vqe_results.max_iteration = self._iter_num
        vqe_results.optimal_value = self._cost_values[-1]
        vqe_results.optimal_parameters = self._optimization_parameters[-1]
        self.ansatz_circuit._update_params(self._optimization_parameters[-1])
        # vqe_results.optimal_circuit = self.ansatz_circuit

        return vqe_results

    def cost_simulator(self, x):
        self._iter_num += 1
        self._optimization_parameters.append(x)
        self.ansatz_circuit._update_params(x)
        eigenvalue = sum(SVSimulator().run(self.ansatz_circuit, hamiltonian=self.hamiltonian)["pauli_expects"])
        self._cost_values.append(eigenvalue)
        return eigenvalue

    def cost_local_qpu(self, x):
        self._iter_num += 1
        self._optimization_parameters.append(x)
        eigenvalue = 0
        for qc, terms in self.measure_circuits:
            qc._update_params(x)
            # qusteed_link = connect('QuafuServer', port=3088, host='10.10.6.242')
            qusteed_link = connect('QuafuServer', port=2223, host='systemq.baqis.ac.cn')
            res, compiled_openqasm, compile_info = qusteed_link.submit(compiler='quafu', input_circuit=qc.to_openqasm(),
                                                                       backend='ScQ-P136', qubits_list=None,
                                                                       optimization_level=None, shots=3000)
            res = {"".join(map(str, key)): value for key, value in res.items()}
            for term in terms:
                eigenvalue += term.coeff * calculate_expectation(sampling_results=res, positions=term.pos)
        self._cost_values.append(eigenvalue)
        print(f"Iteration {self._iter_num}, Energy: {eigenvalue}")
        return eigenvalue

    def cost_general_backend(self, x):
        self._iter_num += 1
        self._optimization_parameters.append(x)
        eigenvalue = 0
        for qc, terms in self.measure_circuits:
            qc._update_params(x)
            res = self.user_backend(qc)
            print('res',res)
            for term in terms:
                eigenvalue += term.coeff * calculate_expectation(sampling_results=res, positions=term.pos)
        self._cost_values.append(eigenvalue)
        return eigenvalue

    # TODO: Check qpu support, may need to change to physical qubits
    def grad_para_qpu(self, x):
        grads = []
        for qc, terms in self.measure_circuits:
            # qc = copy.deepcopy(self.ansatz_circuit)
            qc._update_params(x)
            ham = Hamiltonian(terms)
            grads.append(grad_para_shift_qpu(qc, ham, backend=self.user_backend))
        grads = [sum(values) for values in zip(*grads)]
        return grads

    def grad_para_simulator(self, x):
        self.ansatz_circuit._update_params(x)
        return grad_para_shift(self.ansatz_circuit, self.hamiltonian)

    # def grad_diff(self, x):
    #     grads = []
    #     for qc, terms in self.measure_circuits:
    #         # qc._update_params(x)
    #         qc = copy.deepcopy(self.ansatz_circuit)
    #         qc._update_params(x)
    #         ham = Hamiltonian(terms)
    #         grads.append(grad_finit_diff(qc, ham))
    #     grads = [sum(values) for values in zip(*grads)]
    #     return grads

    def grad_diff(self, x):
        self.ansatz_circuit._update_params(x)
        return grad_finit_diff(self.ansatz_circuit, self.hamiltonian)

    # def grad_qpu(self, x, ansatz_circuit, hamil, backend):
    #     ps = ParamShift()
    #     ansatz_circuit._update_params(x)
    #     return grad_adjoint(ansatz_circuit, hamil)

    # def grad(self, x, ansatz_circuit, hamil, backend):
    #     ansatz_circuit._update_params(x)
    #     return grad_adjoint(ansatz_circuit, hamil)

    def grad_reverse(self, x):
        self.ansatz_circuit._update_params(x)
        return grad_adjoint(self.ansatz_circuit, self.hamiltonian)

    # def grad(self, x):
    #     grads = []
    #     for qc, terms in self.measure_circuits:
    #         # qc._update_params(x)
    #         qc = copy.deepcopy(self.ansatz_circuit)
    #         qc._update_params(x)
    #         ham = Hamiltonian(terms)
    #         grads.append(grad_adjoint(qc, ham))
    #     grads = [sum(values) for values in zip(*grads)]
    #     return grads
import os
import pickle

from qiskit.circuit import QuantumCircuit


class VQAResult:
    """Variational Quantum Algorithms Result."""

    def __init__(self):
        # self._initial_parameters = None
        self.initial_parameters = list
        self.optimization_time = float
        self.optimization_parameters = list
        self.cost_values = list
        self.max_iteration = int
        self.optimal_value = float
        self.optimal_parameters = list
        self.error = None
        # self.optimal_circuit = QuantumCircuit()


def _get_path():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    target_dir = os.path.join(current_dir, '..', '..', "log")
    normalized_path = os.path.normpath(target_dir)
    folder_path = os.path.join(normalized_path, "vqa_results")

    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    return folder_path


def save_results(result, filename: str = None):
    path = _get_path()
    file = os.path.join(path, filename + ".pkl")
    with open(file, 'wb') as f:
        pickle.dump(result, f)

def load_results(filename: str = None) -> VQAResult:
    path = _get_path()
    file = os.path.join(path, filename + ".pkl")
    with open(file, 'rb') as f:
        return pickle.load(f)
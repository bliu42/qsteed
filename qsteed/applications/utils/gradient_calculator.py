import numpy as np
from quafu import QuantumCircuit
from quafu.simulators.simulator import SVSimulator
from quafu.elements import Parameter, ParameterExpression
from quafu import simulate
from qusteed.applications.utils.public import calculate_expectation

def assemble_grads(para_grads, gate_grads):
    grads = []
    for var in para_grads:
        grad_p = para_grads[var]
        fullgrad = 0.
        for pos_g in grad_p:
            pos, gp = pos_g
            gg = gate_grads[pos[0]][pos[1]]
            fullgrad += gg * gp
        grads.append(fullgrad)
    return grads


def grad_para_shift_qpu(qc: QuantumCircuit, hamiltonian, backend):
    """
    Parameter shift gradients. Each gate must have one parameter
    """
    para_grads = qc._calc_parameter_grads()
    gate_grads = [[] for _ in qc.gates]

    for i, op in enumerate(qc.gates):
        if len(op.paras) > 0:
            if isinstance(op.paras[0], Parameter) or isinstance(op.paras[0], ParameterExpression):
                if op.name not in ["RX", "RY", "RZ"]:
                    raise ValueError(
                        "It seems the circuit can not apply parameter-shift rule to calculate gradient.You may need compile the circuit first")

                if backend == "state-simulator":
                    op.paras[0] = op.paras[0] + np.pi / 2
                    res1 = sum(SVSimulator().run(qc, hamiltonian=hamiltonian)["pauli_expects"])
                    op.paras[0] = op.paras[0] - np.pi
                    res2 = sum(SVSimulator().run(qc, hamiltonian=hamiltonian)["pauli_expects"])
                    op.paras[0]._undo(2)
                    gate_grads[i].append((res1 - res2) / 2.)
                # TODO: sampling simulator or real QPU, qc is the real measurement circuit, not the ansatz_circuit
                elif backend == "sampling-simulator":
                    op.paras[0] = op.paras[0] + np.pi / 2
                    sim1 = simulate(qc, shots=3000).counts
                    res1 = 0
                    for term in hamiltonian.paulis:
                        res1 += term.coeff * calculate_expectation(sampling_results=sim1, positions=term.pos)

                    op.paras[0] = op.paras[0] - np.pi
                    sim2 = simulate(qc, shots=3000).counts
                    res2 = 0
                    for term in hamiltonian.paulis:
                        res2 += term.coeff * calculate_expectation(sampling_results=sim2, positions=term.pos)

                    op.paras[0]._undo(2)
                    gate_grads[i].append((res1 - res2) / 2.)
                elif not isinstance(backend, str) and callable(backend):
                    op.paras[0] = op.paras[0] + np.pi / 2
                    sim1 = backend(qc, shots=3000)
                    res1 = 0
                    for term in hamiltonian.paulis:
                        res1 += term.coeff * calculate_expectation(sampling_results=sim1, positions=term.pos)

                    op.paras[0] = op.paras[0] - np.pi
                    sim2 = backend(qc, shots=3000)
                    res2 = 0
                    for term in hamiltonian.paulis:
                        res2 += term.coeff * calculate_expectation(sampling_results=sim2, positions=term.pos)

                    op.paras[0]._undo(2)
                    gate_grads[i].append((res1 - res2) / 2.)

    return assemble_grads(para_grads, gate_grads)


def grad_para_shift(qc: QuantumCircuit, hamiltonian, backend=SVSimulator()):
    """
    Parameter shift gradients. Each gate must have one parameter
    """
    para_grads = qc._calc_parameter_grads()
    gate_grads = [[] for _ in qc.gates]

    for i, op in enumerate(qc.gates):
        if len(op.paras) > 0:
            if isinstance(op.paras[0], Parameter) or isinstance(op.paras[0], ParameterExpression):
                if op.name not in ["RX", "RY", "RZ"]:
                    raise ValueError(
                        "It seems the circuit can not apply parameter-shift rule to calculate gradient.You may need compile the circuit first")
                op.paras[0] = op.paras[0] + np.pi / 2
                res1 = sum(backend.run(qc, hamiltonian=hamiltonian)["pauli_expects"])
                op.paras[0] = op.paras[0] - np.pi
                res2 = sum(backend.run(qc, hamiltonian=hamiltonian)["pauli_expects"])
                op.paras[0]._undo(2)
                gate_grads[i].append((res1 - res2) / 2.)
    return assemble_grads(para_grads, gate_grads)


def grad_finit_diff(qc, hamiltonian, backend=SVSimulator()):
    variables = qc.variables
    print('variables', variables)
    grads = []
    for v in variables:
        v.value += 1e-10
        res1 = sum(backend.run(qc, hamiltonian=hamiltonian)["pauli_expects"])
        v.value -= 2 * 1e-10
        res2 = sum(backend.run(qc, hamiltonian=hamiltonian)["pauli_expects"])
        v.value += 1e-10
        grads.append((res1 - res2) / (2 * 1e-10))

    return grads


def spsa(loss_function, theta, num_iterations=1000, learning_rate=0.1, perturbation=0.05):
    """
    A method for minimizing a loss function via simultaneous perturbation     stochastic approximation (SPSA).

    Parameters:
    loss_function (function): Loss function to be minimized.
    theta (numpy array): The initial values of the parameters.
    learning_rate (float): Size of the step when updating parameters.
    perturbation (float): The noise parameter for randomly generating perturbations.
    num_iterations (int): Number of iterations in the algorithm.

    Returns:
    tuple: A tuple containing the best parameter values found and the
    corresponding loss value.
    """
    # Initialize the best parameter values and loss
    best_theta = theta
    best_loss = loss_function(theta)

    # Initialize the iteration counter
    i = 1

    # Repeat until convergence or the maximum number of iterations is reached
    while i < num_iterations:
        # learning_rate, perturbation with attenuation
        # an = 0.1 / np.power(i + 1, 0.1)
        # cn = 0.1 / np.power(i + 1, 0.1)
        # learning_rate=an
        # perturbation=cn

        # Generate a random perturbation vector with elements that are either +1 or -1
        delta = np.random.choice([-1, 1], size=len(theta))

        # Evaluate the objective function at the positive and negative perturbations
        loss_plus = loss_function(theta + perturbation * delta)
        loss_minus = loss_function(theta - perturbation * delta)

        # Calculate the gradient estimate using the perturbations
        gradient = (loss_plus - loss_minus) / (2 * perturbation * delta)

        # Update the parameter values
        theta = theta - learning_rate * gradient

        # If the new parameter values result in a lower loss, update the best values
        new_loss = loss_function(theta)
        if new_loss < best_loss:
            best_theta = theta
            best_loss = new_loss

        # Increment the iteration counter
        i += 1

    # Return the best parameter values and the corresponding loss value
    return (best_theta, best_loss)
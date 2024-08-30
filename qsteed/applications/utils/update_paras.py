from quafu.elements import Parameter, ParameterExpression
from quafu import QuantumCircuit
import copy


def calculate_expression(paras, variables):
    """
    计算变量的表达式
    """
    def evaluate(operand):
        if isinstance(operand, (int, float)):
            return operand
        if isinstance(operand, Parameter):
            variable = [item for item in variables if item.name == operand.name]
            return variable[0].value
            # return operand
        elif isinstance(operand, ParameterExpression):
            return calculate_expression(operand, variables)

    # result = paras.pivot
    # for func, operand in zip(paras.funcs, paras.operands):
    #     result = func(result, evaluate(operand))
    result = paras.pivot
    if isinstance(result, Parameter):
        result = [item for item in variables if item.name == result.name][0].value
    for func, operand in zip(paras.funcs, paras.operands):
        result = func(result, evaluate(operand))
    return result


from collections import defaultdict
from quafu.elements.element_gates import RXGate, RYGate, RZGate

def order_variables_indices(circuit: QuantumCircuit, initial_variables: list=None):
    circuit.get_parameter_grads()
    transpiled_variables = circuit._variables
    initial_order = {item.name: i for i, item in enumerate(initial_variables)}
    order_indices = [initial_order[x.name] for x in transpiled_variables]
    return order_indices

def update_circuit_paras(circuit: QuantumCircuit, parameters=None, order_variables=None):

    # circuit.get_parameter_grads()
    # transpiled_variables = circuit._variables
    # initial_order = {item.name: i for i, item in enumerate(initial_variables)}
    # transpiled_order_indices = [initial_order[x.name] for x in transpiled_variables]

    copy_circuit = copy.deepcopy(circuit)
    copy_circuit._update_params(parameters, order=order_variables)

    variables = copy.deepcopy(copy_circuit.variables)

    update_circuit = QuantumCircuit(copy_circuit.num)
    for gate in copy_circuit.gates:
        if len(gate.paras) == 0:
            update_circuit.add_gate(gate)
        elif not isinstance(gate.paras[0], (Parameter, ParameterExpression)):
            update_circuit.add_gate(gate)
        else:
            paras = calculate_expression(gate.paras[0], variables)
            pos = gate.pos
            if gate.name == 'RX':
                update_circuit.add_gate(RXGate(pos[0], paras))
            elif gate.name == 'RY':
                update_circuit.add_gate(RYGate(pos[0], paras))
            elif gate.name == 'RZ':
                update_circuit.add_gate(RZGate(pos[0], paras))
            else:
                raise TypeError("Currently only Rx, Ry and Rz gates are supported.")
    update_circuit.measures = circuit.measures
    return update_circuit

# parameters=[0,0,0,0,3,3,3,3,3,3]
# initial_variables = compiler.model.datadict['variables']
# order_indices = order_variables_indices(compiled_circuit, initial_variables)
# print('order_indices',order_indices)
# update_circuit, variables=get_update_circuit(compiled_circuit, parameters=parameters,order_variables=order_indices)
# update_circuit.draw_circuit()
#
# import numpy as np
# sorted_tuples = sorted(zip(order_indices, variables), key=lambda x: x[0])
# theta_new = [elem for _, elem in sorted_tuples]
# print(np.sin(theta_new[1]) - 4. * theta_new[0] + theta_new[2]*theta_new[0])
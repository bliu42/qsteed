from quafu.algorithms.hamiltonian import Hamiltonian


def group_hamiltonian_terms(hamiltonian: Hamiltonian):
    """
    Group Hamiltonian terms according to measurement basis

    Args:
    - hamiltonian (Hamiltonian): The Hamiltonian terms, represented as strings.

    Returns:
    - dict: A dictionary with keys representing measurement bases or specific cases, and values being lists of terms.
    """
    groups = {'X': [], 'Y': [], 'Z': [], 'Mixed': []}

    for term in hamiltonian.paulis:
        x_count = term.paulistr.count('X')
        y_count = term.paulistr.count('Y')
        z_count = term.paulistr.count('Z')

        if x_count > 0 and y_count == 0 and z_count == 0:
            groups['X'].append(term)
        elif y_count > 0 and x_count == 0 and z_count == 0:
            groups['Y'].append(term)
        elif z_count > 0 and x_count == 0 and y_count == 0:
            groups['Z'].append(term)
        else:
            groups['Mixed'].append(term)

    groups = {key: value for key, value in groups.items() if value}

    return groups


def calculate_expectation(sampling_results: dict = None, positions: list = None):
    """
    Calculate the expectation value based on specified qubit positions with Z operators.

    Args::
        sampling_results (dict): Measurement results, keys are bitstrings and values are counts or probabilities.
        positions (list of int): List of qubit positions that Z operators are applied to.

    Returns:
        expectation (float): The expectation value for the specified positions.
    """
    total_samples = sum(sampling_results.values())
    expectation_value = 0

    # Loop through each measurement result
    for bitstring, count in sampling_results.items():
        # Initialize the measurement value for this term
        measurement_value = 1

        # Apply Z operations based on the specified positions
        for pos in positions:
            bit = int(bitstring[pos])
            measurement_value *= 1 if bit == 0 else -1

        # Update the expectation value with this measurement's contribution
        expectation_value += measurement_value * count

    expectation = expectation_value / total_samples

    return expectation
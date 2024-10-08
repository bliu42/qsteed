# This code is part of QSteed.
#
# (C) Copyright 2024 Beijing Academy of Quantum Information Sciences
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from math import pi
from typing import List

import numpy as np
from quafu.elements import Instruction
from quafu.elements.element_gates import MCRXGate, CRXGate, HGate, ToffoliGate, CPGate, RYGate, SGate, SdgGate, CXGate

from qsteed.passes.basepass import UnrollPass


class MCRXToCNOT(UnrollPass):
    """
    The MCRXtoCNOT pass.
    convert MCRX gate to {CNOT, H, Rz}.

    MCRX gate decomposition rule :
    (control bit 0:1, target bit: 3)
    q[0]  --*--          q[0]  ------------------------*---------------------*----*-----T-----*---------------------------*---------------------*----*----T------*------
            |                                          |                     |    |           |                           |                     |    |           |
    q[1]  --*--  ≡       q[1]  ------------*-----------|---------*-----T-----|----+----Tdg----+---------------*-----------|---------*----T------|----+----Tdg----+------
            |                              |           |         |           |                                |           |         |           |
    q[2]  --RX(θ)--      q[2]  --S----H----+----Tdg----+----T----+----Tdg----+----T-----H----RY(-θ/2)----H----+----Tdg----+----T----+----Tdg----+----T----H----RY(θ/2)--

    (control bit 0:2, target bit: 3)
    q[0]  --*--          q[0]  --------------------------------------+----*-----+----*----+-----*----+----*-----------------------------------------------------+----*-----+----*----+-----*----+----*--------------------------
            |                                                        |    |     |    |    |     |    |    |                                                     |    |     |    |    |     |    |    |
    q[1]  --*--  ≡       q[1]  ------------------+----*----+----*----*----|-----|----|----*-----|----|----|---------------------------------+----*----+----*----*----|-----|----|----*-----|----|----|--------------------------
            |                                    |    |    |    |         |     |    |          |    |    |                                 |    |    |    |         |     |    |          |    |    |
    q[2]  --*--          q[2]  ------------*-----*----|----*----|---------|-----*----|----------|----*----|---------------------------*-----*----|----*----|---------|-----*----|----------|----*----|--------------------------
            |                              |          |         |         |          |          |         |                           |          |         |         |          |          |         |
    q[3]  --RX(θ)--      q[3]  --S----H----R1--------R1_--------R1--------R1_--------R1--------R1_--------R1----H----RY(-θ/2)----H----R1--------R1_--------R1--------R1_--------R1--------R1_--------R1----H----RY(θ/2)----Sdg--
    """

    def __init__(self) -> None:
        super().__init__()
        self.original = MCRXGate([0, 1], 2, 0).name.lower()
        self.basis = [CXGate.name.lower(), HGate.name.lower(), CPGate(0, 1, 0).name.lower(),
                      ToffoliGate(0, 1, 2).name.lower(), RYGate(0, 0).name.lower(), SGate.name.lower(),
                      SdgGate.name.lower()]

    def run(self, op: Instruction) -> List[Instruction]:
        rule = []
        if isinstance(op, MCRXGate):
            control_bits = op.ctrls
            if isinstance(op.paras, list):
                theta = op.paras[0]
            else:
                theta = op.paras
            if isinstance(op.targs, list):
                target_bit = op.targs[0]
            else:
                target_bit = op.targs

            if len(control_bits) == 1:
                rule.append(CRXGate(control_bits[0], target_bit))
            elif len(control_bits) == 2:
                rule.append(SGate(target_bit))
                rule.append(ToffoliGate(control_bits[0], control_bits[1], target_bit))
                rule.append(RYGate(target_bit, -theta / 2))
                rule.append(ToffoliGate(control_bits[0], control_bits[1], target_bit))
                rule.append(RYGate(target_bit, theta / 2))
                rule.append(SdgGate(target_bit))
            else:
                graycode = self.build_gray_code(len(control_bits))
                theta = pi / 2 ** (len(control_bits) - 1)
                last_code = graycode[0, :]
                rule.append(SGate(target_bit))
                rule.append(HGate(target_bit))
                # loop the gray code based on the line order
                for i in range(1, len(graycode)):
                    code = graycode[i, :]
                    ones_bits = np.where(code == 1)[0]
                    set_idx = ones_bits[0]
                    diff_idx = np.where(code != last_code)[0][0]
                    if diff_idx != set_idx:
                        rule.append(CXGate(control_bits[diff_idx], control_bits[set_idx]))
                    elif len(ones_bits) >= 2:
                        next_idx = ones_bits[1]
                        rule.append(CXGate(control_bits[next_idx], control_bits[set_idx]))
                    if np.sum(code) % 2 == 0:
                        rule.append(CPGate(control_bits[set_idx], target_bit, -theta))
                    else:
                        rule.append(CPGate(control_bits[set_idx], target_bit, theta))
                    last_code = code
                rule.append(HGate(target_bit))
                rule.append(RYGate(target_bit, -theta / 2))
                rule.append(HGate(target_bit))
                last_code = graycode[0, :]
                for i in range(1, len(graycode)):
                    code = graycode[i, :]
                    ones_bits = np.where(code == 1)[0]
                    set_idx = ones_bits[0]
                    diff_idx = np.where(code != last_code)[0][0]
                    if diff_idx != set_idx:
                        rule.append(CXGate(control_bits[diff_idx], control_bits[set_idx]))
                    elif len(ones_bits) >= 2:
                        next_idx = ones_bits[1]
                        rule.append(CXGate(control_bits[next_idx], control_bits[set_idx]))
                    if np.sum(code) % 2 == 0:
                        rule.append(CPGate(control_bits[set_idx], target_bit, -theta))
                    else:
                        rule.append(CPGate(control_bits[set_idx], target_bit, theta))
                    last_code = code
                rule.append(HGate(target_bit))
                rule.append(RYGate(target_bit, theta / 2))
                rule.append(SdgGate(target_bit))
        else:
            rule.append(op)
        self.rule = rule
        return rule

    # generate the gray code to assign control gate between control bits
    def build_gray_code(self, n):
        code = np.full((2 ** n, n), 0)
        block = np.repeat([0, 1], 2 ** (n - 1))
        code[:, 0] = block
        for i in range(1, n):
            block = np.repeat([0, 1, 1, 0], 2 ** (n - 1 - i))
            print(block)
            code[:, i] = np.tile(block, 2 ** (i - 1))
        return code

# Copyright 2026 Santos C. Borom
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
########################################################################
"""
tests.cirq_ionq_api
Tests basic interaction with IonQ's API via the Cirq-IonQ interface, 
primarily using IonQ's backend simulators (like Forte-1).
"""

import cirq
import cirq_ionq as ionq
import os

API_KEY = os.environ.get('IONQ_API_KEY')
if not API_KEY:
    raise ValueError("IONQ_API_KEY environment variable not set!")

# A simple circuit
qubit = cirq.LineQubit(0)
circuit = cirq.Circuit(
    cirq.X(qubit) ** 0.5,   # √X gate
    cirq.measure(qubit, key='x')
)

service = ionq.Service(api_key=API_KEY)

# === Forte-1 Noisy Simulator ===
result = service.run(
    circuit=circuit,
    repetitions=100,
    target='simulator',                    # Must be simulator
    extra_query_params={
        "noise": {"model": "forte-1"}
    }
)

histogram = result.histogram(key='x')
print(f'Histogram: {histogram}')
print(f'Data:\n{result.data}')
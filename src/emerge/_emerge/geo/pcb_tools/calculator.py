# EMerge is an open source Python based FEM EM simulation module.
# Copyright (C) 2025  Robert Fennis.

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, see
# <https://www.gnu.org/licenses/>.

import numpy as np
from emsutil import Material
from ...const import Z0 as n0

PI = np.pi
TAU = 2*PI

def microstrip_z0(W: float, th: float, er: float):
    u = W/th
    fu = 6 + (TAU - 6)*np.exp(-((30.666/u)**(0.7528)))
    Z0 = n0/(TAU*np.sqrt(er))* np.log(fu/u + np.sqrt(1+(2/u)**2))
    return Z0


class PCBCalculator:

    def __init__(self, layers: np.ndarray, materials: list[Material], unit: float):
        self.layers: np.ndarray = layers
        self.mat: list[Material] = materials
        self.unit = unit

    def z0(self, Z0: float, layer: int = -1, ground_layer: int = 0, f0: float = 1e9, er: float = None) -> float:
        """Computes the width of a microstrip line for a given characteristic impedance Z0.

        Args:
            Z0 (float): The desired characteristic impedance
            layer (int, optional): The microstripline layer. Defaults to -1 (top).
            ground_layer (int, optional): The relative ground later. Defaults to 0 (bottom).
            f0 (float, optional): The frequency at which to evaluate the dielectric constant. Defaults to 1e9.
            er (float, optional): The dielectric permittivity to use instead of a value based on material layers. Defaults to None.

        Returns:
            float: The width of the microstrip line in the PCB unit.
        """
        layer = layer % len(self.layers)
        ground_layer = ground_layer % len(self.layers)
        i1 = min(layer, ground_layer)
        i2 = max(layer, ground_layer)
        mats = [self.mat[i] for i in np.arange(0, len(self.layers))[i1:i2]]
        ers = [mat.er.scalar(f0) for mat in mats]
        ths = [self.layers[i] for i in np.arange(0, len(self.layers))[i1:i2]]
        th = abs(self.layers[layer] - self.layers[ground_layer])*self.unit
        mean_er = sum([ers[i]*ths[i] for i in range(len(ers))])/sum(ths)
        
        ws = np.geomspace(1e-6,1e-1,101)
        if er is None:
            er = mean_er
        Z0ms = microstrip_z0(ws, th, er)
        return np.interp(Z0, Z0ms[::-1], ws[::-1])/self.unit

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


from .mesh3d import Mesh3D
from .geometry import GeoObject, _GeometryManager, _GEOMANAGER
from .dataset import SimulationDataset
from loguru import logger

from typing import Any


class SimState:
    
    def __init__(self):
        self.mesh: Mesh3D = Mesh3D()
        self.geos: list[GeoObject] = []
        self.data: SimulationDataset = SimulationDataset()
        self.params: dict[str, float] = dict()
        self._stashed: SimulationDataset | None = None
        self.manager: _GeometryManager = _GEOMANAGER
    
    @property
    def current_geo_state(self) -> list[GeoObject]:
        return self.manager.all_geometries()
    
    def reset_geostate(self, modelname: str) -> None:
        _GEOMANAGER.reset(modelname)
        self.clear_mesh()
        
    def init(self, modelname: str) -> None:
        self.mesh = Mesh3D()
        self.geos = []
        self.reset_geostate(modelname)
        self.init_data()
        
    def stash(self) -> None:
        self._stashed = self.data
        self.data = SimulationDataset()
    
    def set_parameters(self, parameters: dict[str, float]) -> None:
        self.params = parameters
        
    def init_data(self) -> None:
        self.data.sim.new(**self.params)
        
    def reload(self) -> SimulationDataset:
        old = self._stashed
        self.data = self._stashed
        self._stashed = None
        return old
    
    def reset_mesh(self) -> None:
        self.mesh = Mesh3D()
        
    def set_mesh(self, mesh: Mesh3D) -> None:
        self.mesh = mesh
        
    def set_geos(self, geos: list[GeoObject]) -> None:
        self.geos = geos
        _GEOMANAGER.set_geometries(geos)

    def clear_mesh(self) -> None:
        self.mesh = Mesh3D()
        
    def store_geometry_data(self) -> None:
        """Saves the current geometry state to the simulatin dataset
        """
        logger.trace('Storing geometries in data.sim')
        self.geos = self.current_geo_state
        self.data.sim['geos'] = self.geos
        self.data.sim['mesh'] = self.mesh
        
    def get_dataset(self) -> dict[str, Any]:
        return dict(simdata=self.data, mesh=self.mesh)
    
    def load_dataset(self, dataset: dict[str, Any]):
        self.data = dataset['simdata']
        self.mesh = dataset['mesh']
        
    def activate(self, _indx: int | None = None, **variables):
        """Searches for the permutaions of parameter sweep variables and sets the current geometry to the provided set."""
        if _indx is not None:
            dataset = self.data.sim.index(_indx)
        else:
            dataset = self.data.sim.find(**variables)
        
        variables = ', '.join([f'{key}={value}' for key,value in dataset.vars.items()])
        logger.info(f'Activated entry with variables: {variables}')
        self.set_mesh(dataset['mesh'])
        self.set_geos(dataset['geos'])
        return self
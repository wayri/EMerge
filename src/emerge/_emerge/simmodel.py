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

from __future__ import annotations
from .mesher import Mesher
from .geometry import GeoObject
from .geo.modeler import Modeler
from .physics.microwave.microwave_3d import Microwave3D
from .mesh3d import Mesh3D
from .selection import Selector, Selection
from .dataset import SimulationDataset
from .logsettings import LOG_CONTROLLER, DEBUG_COLLECTOR
from .plot.pyvista import PVDisplay
from .periodic import PeriodicCell
from .cacherun import get_build_section, get_run_section
from .settings import DEFAULT_SETTINGS, Settings
from .solver import EMSolver, Solver
from .simstate import SimState
from typing import Literal, Generator, Any
from loguru import logger
import numpy as np
import gmsh # type: ignore
import os
import inspect
from pathlib import Path
import joblib
from atexit import register
import signal
from .. import __version__
from .mesher import Algorithm3D
############################################################
#                   EXCEPTION DEFINITIONS                  #
############################################################

_GMSH_ERROR_TEXT = """
--------------------------
Known problems/solutions:
(1) - PLC Error:  A segment and a facet intersect at point
    This can be caused when approximating thin curved volumes. Try to decrease the mesh size for that region.
--------------------------
"""

class SimulationError(Exception):
    pass

class VersionError(Exception):
    pass

############################################################
#                 BASE 3D SIMULATION MODEL                 #
############################################################


class Simulation:

    def __init__(self, 
                 modelname: str, 
                 loglevel: Literal['TRACE','DEBUG','INFO','WARNING','ERROR'] = 'INFO',
                 load_file: bool = False,
                 save_file: bool = False,
                 write_log: bool = False,
                 path_suffix: str = ".EMResults"):
        """Generate a Simulation class object.

        As a minimum a file name should be provided. Additionally you may provide it with any
        class that inherits from BaseDisplay. This will then be used for geometry displaying.

        Args:
            modelname (str): The name of the simulation model. This will be used for filenames and path names when saving data.
            loglevel ("DEBUG","INFO","WARNING","ERROR", optional): The loglevel to use for loguru. Defaults to 'INFO'.
            load_file (bool, optional): If the simulatio model should be loaded from a file. Defaults to False.
            save_file (bool, optional): if the simulation file should be stored to a file. Defaults to False.
            write_log (bool, optional): If a file should be created that contains the entire log of the simulation. Defaults to False.
            path_suffix (str, optional): The suffix that will be added to the results directory. Defaults to ".EMResults".
        """

        caller_file = Path(inspect.stack()[1].filename).resolve()
        base_path = caller_file.parent

        self.modelname = modelname
        self.modelpath = base_path / (modelname.lower()+path_suffix)
        self.mesher: Mesher = Mesher()
        self.modeler: Modeler = Modeler()
        
        self.state: SimState = SimState()
        self.select: Selector = Selector()
        self.settings: Settings = DEFAULT_SETTINGS

        ## Display
        self.display: PVDisplay = PVDisplay(self.state)
        
        ## STATES
        self.__active: bool = False
        self._defined_geometries: bool = False
        self._cell: PeriodicCell | None = None
        self.save_file: bool = save_file
        self.load_file: bool = load_file
        self._cache_run: bool = False
        self._file_lines: str = ''
        
        ## Physics
        self.mw: Microwave3D = Microwave3D(self.state, self.mesher, self.settings)

        self._initialize_simulation()

        self.set_loglevel(loglevel)
        
        if write_log:
            self.set_write_log()

        LOG_CONTROLLER._flush_log_buffer()
        LOG_CONTROLLER._sys_info()

        self.__post_init__()

    ############################################################
    #                       PRIVATE FUNCTIONS                  #
    ############################################################

    @property
    def data(self) -> SimulationDataset:
        return self.state.data
    
    @property
    def mesh(self) -> Mesh3D:
        return self.state.mesh
    
    def __post_init__(self):
        pass

    def __setitem__(self, name: str, value: Any) -> None:
        """Store data in the current data container"""
        self.data.sim[name] = value

    def __getitem__(self, name: str) -> Any:
        """Get the data from the current data container"""
        return self.data.sim[name]
    
    def __enter__(self) -> Simulation:
        """This method is depricated with the new atexit system. It still exists for backwards compatibility.

        Returns:
            Simulation: the Simulation object
        """
        return self

    def __exit__(self, type, value, tb):
        """This method no longer does something. It only serves as backwards compatibility."""
        self._exit_gmsh()
        return False
    
    def _install_signal_handlers(self):
        # on SIGINT (Ctrl-C) or SIGTERM, call our exit routine
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._handle_signal)
    
    def _handle_signal(self, signum, frame):
        """
        Signal handler: do our cleanup, then re-raise
        the default handler so that exit code / traceback
        is as expected.
        """
        try:
            # run your atexit-style cleanup
            self._exit_gmsh()
        except Exception:
            # log but don’t block shutdown
            logger.exception("Error during signal cleanup")
        finally:
            # restore default handler and re‐send the signal
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

    def _initialize_simulation(self):
        """Initializes the Simulation data and GMSH API with proper shutdown routines.
        """
        self.state.init(self.modelname)
        
        # If GMSH is not yet initialized (Two simulation in a file)
        if gmsh.isInitialized() == 0:
            logger.debug('Initializing GMSH')
            
            gmsh.initialize()
            # Set an exit handler for Ctrl+C cases
            self._install_signal_handlers()

            # Restier the Exit GMSH function on proper program abortion
            register(self._exit_gmsh)
        else:
            gmsh.finalize()
            gmsh.initialize()
            
        # Create a new GMSH model or load it
        if not self.load_file:
            gmsh.model.add(self.modelname)
        else:
            self.load()

        # Set the Simulation state to active
        self.__active = True
        return self

    def _exit_gmsh(self):
        # If the simulation object state is still active (GMSH is running)
        if not self.__active:
            return
        logger.debug('Exiting program')
        
        if DEBUG_COLLECTOR.any_warnings:
            logger.warning('EMerge simulation warnings:')
        for i, report in DEBUG_COLLECTOR.all_reports():
            logger.warning(f'{i}: {report}')
        # Save the file first
        if self.save_file:
            self.save()
        # Finalize GMSH
        if gmsh.isInitialized():
            gmsh.finalize()

        logger.debug('GMSH Shut down successful')
        # set the state to active
        self.__active = False
        
    
    ############################################################
    #                       PUBLIC FUNCTIONS                  #
    ############################################################

    def cache_build(self) -> bool:
        """Checks if all the lines inside this if statement block are the same as those
        stored from a previous run. If so, then it returns false. Else it returns True.
        
        Can be used to capture an entire model simulation.
        
        Example:
        
        >>> if model.cache_build():
        >>>     box = em.geo.Box(...)
        >>>     # Other lines
        >>>     model.mw.run_sweep()
        >>> data = model.data.mw

        Returns:
            bool: If the code is not the same
        """
        
        self.save_file = True
        self._cache_run = True
        filestr = get_build_section()
        self._file_lines = filestr
        cachepath = self.modelpath / 'pylines.txt'
        
        # If there is not pylines file, simulate (can't have been run).
        if not cachepath.exists():
            logger.info('No cached data detected, running file')
            return True
        
        with open(cachepath, 'r') as file:
            lines = file.read()
        
        if lines==filestr:
            logger.info('Cached data detected! Loading data!')
            self.load()
            return False
        logger.info('Different cached data detected, rebuilding file.')
        return True
    
    def cache_run(self) -> bool:
        """Checks if all the lines before this call are the same as the lines
        stored from a previous run. If so, then it returns false. Else it returns True.
        
        Can be used to capture a run_sweep() call.
        
        Example:
        
        >>> if model.cache_run():
        >>>     model.mw.run_sweep()
        >>> data = model.data.mw

        Returns:
            bool: If the code is not the same
        """
        self.save_file = True
        self._cache_run = True
        filestr = get_run_section()
        self._file_lines = filestr
        cachepath = self.modelpath / 'pylines.txt'
        
        # If there is not pylines file, simulate (can't have been run).
        if not cachepath.exists():
            logger.info('No cached data detected, running simulation!')
            return True
        
        with open(cachepath, 'r') as file:
            lines = file.read()
        
        if lines==filestr:
            logger.info('Cached data detected! Loading data!')
            self.load()
            return False
        logger.info('Different cached data detected, rerunning simulation.')
        return True
    
    def check_version(self, target_version: str, *, log: bool = False) -> None:
        """
        Ensure the script targets an EMerge version compatible with the current runtime.

        Parameters
        ----------
        target_version : str
            The EMerge version this script was written for (e.g. "1.4.0").
        log : bool, optional
            If True and a `logger` is available, emit a single WARNING with the same
            message as the exception. Defaults to False.

        Raises
        ------
        VersionError
            If the script's target version differs from the running EMerge version.
        """
        try:
            from packaging.version import Version as _V
            v_script = _V(target_version)
            v_runtime = _V(__version__)
            newer = v_script > v_runtime
            older = v_script < v_runtime
        except Exception:
            def _parse(v: str):
                try:
                    return tuple(int(p) for p in v.split("."))
                except Exception:
                    # Last-resort: compare as strings to avoid crashing the check itself
                    return tuple(v.split("."))
            v_script = _parse(target_version)
            v_runtime = _parse(__version__)
            newer = v_script > v_runtime
            older = v_script < v_runtime

        if not newer and not older:
            return  # exact match

        if newer:
            msg = (
                f"Script targets EMerge {target_version}, but runtime is {__version__}. "
                "The script may rely on features added after your installed version. "
                "Recommended: upgrade EMerge (`pip install --upgrade emerge`). "
                "If you know the script is compatible, you may remove this check."
            )
        else:  # older
            msg = (
                f"Script targets EMerge {target_version}, but runtime is {__version__}. "
                "APIs may have changed since the targeted version. "
                "Recommended: update the script for the current EMerge, or run a matching older release. "
                "If you know the script is compatible, you may remove this check."
            )

        if log:
            try:
                logger.warning(msg)
            except Exception:
                pass

        raise VersionError(msg)

    def activate(self, _indx: int | None = None, **variables) -> Simulation:
        """Searches for the permutaions of parameter sweep variables and sets the current geometry to the provided set."""
        self.state.activate(_indx, **variables)
        return self
    
    def save(self) -> None:
        """Saves the current model in the provided project directory."""
        # Ensure directory exists
        if not self.modelpath.exists():
            self.modelpath.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {self.modelpath}")

        # Save mesh
        mesh_path = self.modelpath / 'mesh.msh'
        brep_path = self.modelpath / 'model.brep'

        gmsh.option.setNumber('Mesh.SaveParametric', 1)
        gmsh.option.setNumber('Mesh.SaveAll', 1)
        gmsh.model.geo.synchronize()
        gmsh.model.occ.synchronize()

        gmsh.write(str(mesh_path))
        gmsh.write(str(brep_path))
        logger.info(f"Saved mesh to: {mesh_path}")

        # Pack and save data
        dataset = self.state.get_dataset()
        data_path = self.modelpath / 'simdata.emerge'
        
        joblib.dump(dataset, str(data_path))
        
        if self._cache_run:
            cachepath = self.modelpath / 'pylines.txt'
            with open(str(cachepath), 'w') as f_out:
                f_out.write(self._file_lines)
            
        logger.info(f"Saved simulation data to: {data_path}")

    def load(self) -> None:
        """Loads the model from the project directory."""
        mesh_path = self.modelpath / 'mesh.msh'
        brep_path = self.modelpath / 'model.brep'
        data_path = self.modelpath / 'simdata.emerge'

        if not mesh_path.exists() or not data_path.exists():
            raise FileNotFoundError("Missing required mesh or data file.")

        # Load GMSH Mesh (Ideally Id remove)
        gmsh.open(str(brep_path))
        gmsh.merge(str(mesh_path))
        gmsh.model.geo.synchronize()
        gmsh.model.occ.synchronize()
        
        logger.info(f"Loaded mesh from: {mesh_path}")
        datapack = joblib.load(str(data_path))
        self.state.load_dataset(datapack)
        self.state.activate(0)
        
        logger.info(f"Loaded simulation data from: {data_path}")
    
    def set_loglevel(self, loglevel: Literal['DEBUG','INFO','WARNING','ERROR']) -> None:
        """Set the loglevel for loguru.

        Args:
            loglevel ('DEBUG','INFO','WARNING','ERROR'): The loglevel
        """
        logger.trace(f'Setting loglevel to {loglevel}')
        LOG_CONTROLLER.set_std_loglevel(loglevel)
        if loglevel not in ('TRACE'):
            gmsh.option.setNumber("General.Terminal", 0)

    def set_write_log(self) -> None:
        """Adds a file output for the logger."""
        logger.trace(f'Writing log to path = {self.modelpath}')
        LOG_CONTROLLER.set_write_file(self.modelpath)
        
    def view(self, 
             selections: list[Selection] | None = None, 
             use_gmsh: bool = False,
             plot_mesh: bool = False,
             volume_mesh: bool = True,
             opacity: float | None = None,
             labels: bool = False) -> None:
        """View the current geometry in either the BaseDisplay object (PVDisplay only) or
        the GMSH viewer.

        Args:
            selections (list[Selection] | None, optional): Optional selections to highlight. Defaults to None.
            use_gmsh (bool, optional): If GMSH's GUI should be used. Defaults to False.
            plot_mesh (bool, optional): If the mesh should be plot instead of the object. Defaults to False.
            volume_mesh (bool, optional): If the internal mesh should be plot instead of only the surface boundary mesh. Defaults to True
            opacity (float | None, optional): The object/mesh opacity. Defaults to None.
            labels: (bool, optional): If geometry name labels should be shown. Defaults to False.
        """
        if not (self.display is not None and self.mesh.defined) or use_gmsh:
            gmsh.model.occ.synchronize()
            gmsh.fltk.run()
            return
        for geo in self.state.current_geo_state:
            self.display.add_object(geo, mesh=plot_mesh, opacity=opacity, volume_mesh=volume_mesh, label=labels)
        if selections:
            [self.display.add_object(sel, color='red', opacity=0.6, label=labels) for sel in selections]
        self.display.show()

        return None
        
    def set_periodic_cell(self, cell: PeriodicCell):
        """Set the given periodic cell object as the simulations peridicity.

        Args:
            cell (PeriodicCell): The PeriodicCell class
        """
        logger.trace(f'Setting {cell} as periodic cell object')
        self.mw.bc._cell = cell
        self._cell = cell

    def set_resolution(self, resolution: float) -> Simulation:
        """Sets the discretization resolution in the various physics interfaces.
        

        Args:
            resolution (float): The resolution as a float. Lower resolution is a finer mesh 

        Returns:
            Simulation: _description_
        """
        self.mw.set_resolution(resolution)
        
    def commit_geometry(self, *geometries: GeoObject | list[GeoObject]) -> None:
        """Finalizes and locks the current geometry state of the simulation.

        The geometries may be provided (legacy behavior) but are automatically managed in the background.
        
        """
        logger.trace('Committing final geometry.')
        self.state.store_geometry_data()
        
        logger.trace(f'Parsed geometries = {self.state.geos}')
        
        self.mesher.submit_objects(self.state.geos)
        
        self._defined_geometries = True
        self.display._facetags = [dt[1] for dt in gmsh.model.get_entities(2)]
    
    def all_geos(self) -> list[GeoObject]:
        """Returns all geometries in a list

        Returns:
            list[GeoObject]: A list of all GeoObjects
        """
        return self.state.current_geo_state
        
    def generate_mesh(self, regenerate: bool = False) -> None:
        """Generate the mesh. 
        This can only be done after commit_geometry(...) is called and if frequencies are defined.

        Args:
            name (str, optional): The mesh file name. Defaults to "meshname.msh".

        Raises:
            ValueError: ValueError if no frequencies are defined.
        """
        logger.info('Starting mesh generation phase.')
        if not regenerate:
            
            if not self._defined_geometries:
                self.commit_geometry()
            
            logger.trace(' (1) Installing periodic boundaries in mesher.')
            # Set the cell periodicity in GMSH
            if self._cell is not None:
                self.mesher.set_periodic_cell(self._cell)
            
            self.mw._initialize_bcs(self.state.manager.get_surfaces())

            # Check if frequencies are defined: TODO: Replace with a more generic check
            if self.mw.frequencies is None:
                raise ValueError('No frequencies defined for the simulation. Please set frequencies before generating the mesh.')

        gmsh.model.occ.synchronize()

        # Set the mesh size
        self.mesher._configure_mesh_size(self.mw.get_discretizer(), self.mw.resolution) # This makes no sense to do this here
            
        # Validity check
        x1, y1, z1, x2, y2, z2 = gmsh.model.getBoundingBox(-1, -1)
        bb_volume = (x2-x1)*(y2-y1)*(z2-z1)
        wl = 299792458/self.mw.frequencies[-1]
        Nelem = int(5 * bb_volume / (wl**3))
        if Nelem > 100_000 and DEFAULT_SETTINGS.size_check:
            DEBUG_COLLECTOR.add_report(f'An estimated {Nelem} tetrahedra are required for the bounding box of the geometry. This may imply a simulation domain that is very large.' + 
                                       'To disable this message. Set the .size_check parameter in model.settings to False.')
            
            raise SimulationError('Simulation requires too many elements.')
        logger.trace(' (2) Calling GMSH mesher')
        try:
            gmsh.logger.start()
            gmsh.model.mesh.generate(3)
            logs = gmsh.logger.get()
            gmsh.logger.stop()
            for log in logs:
                logger.trace('[GMSH] ' + log)
        except Exception:
            logger.error('GMSH Mesh error detected.')
            print(_GMSH_ERROR_TEXT)
            raise
        
        logger.info('GMSH Meshing complete!')
        self.mesh._pre_update(self.mesher._get_periodic_bcs())
        self.mesh.exterior_face_tags = self.mesher.domain_boundary_face_tags
        gmsh.model.occ.synchronize()
        logger.trace(' (3) Mesh routine complete')
        
    def parameter_sweep(self, clear_mesh: bool = True, **parameters: np.ndarray) -> Generator[tuple[float,...], None, None]:
        """Executes a parameteric sweep iteration.

        The first argument clear_mesh determines if the mesh should be cleared and rebuild in between sweeps. This is usually needed
        except when you change only port-properties or material properties. The parameters of the sweep can be provided as a set of 
        keyword arguments. As an example if you defined the axes: width=np.linspace(...) and height=np.linspace(...). You can
        add them as arguments using .parameter_sweep(True, width=width, height=height).

        The rest of the simulation commands should be inside the iteration scope

        Args:
            clear_mesh (bool, optional): Wether to clear the mesh in between sweeps. Defaults to True.

        Yields:
            Generator[tuple[float,...], None, None]: The parameters provided

        Example:
         >>> for W, H in model.parameter_sweep(True, width=widths, height=heights):
         >>>    // build simulation
         >>>    data = model.run_sweep()
         >>> // Extract the data
         >>> widths, heights, frequencies, S21 = data.ax('width','height','freq').S(2,1)
        """
        paramlist = sorted(list(parameters.keys()))
        dims = np.meshgrid(*[parameters[key] for key in paramlist], indexing='ij')
        dims_flat = [dim.flatten() for dim in dims]
        
        self.mw.cache_matrices = False
        logger.trace('Starting parameter sweep.')
        
        for i_iter in range(dims_flat[0].shape[0]):
            
            if clear_mesh and i_iter > 0:
                logger.info('Cleaning up mesh.')
                gmsh.clear()
                self.state.reset_geostate(self.modelname)
                self.mw.reset()
                
            
            params = {key: dim[i_iter] for key,dim in zip(paramlist, dims_flat)}
            
            self.state.set_parameters(params)
            
            logger.info(f'Iterating: {params}')
            if len(dims_flat)==1:
                yield dims_flat[0][i_iter]
            else:
                yield (dim[i_iter] for dim in dims_flat) # type: ignore
            
            
            if not clear_mesh:
                self.state.store_geometry_data()
        
        if not clear_mesh:
            self.state.store_geometry_data()
        
        self.mw.cache_matrices = True
        
        
    def export(self, filename: str):
        """Exports the model or mesh depending on the extension. 
        
        Exporting is realized by GMSH.
        Supported file formats are:
        
        3D Model: .opt, .geo_unrolled, .brep, .xao ,.step and .iges
        Mesh: .msh, .inp, .key, ._0000.rad, .celum, .cgns, .diff, .unv, .ir3, .mes, .mesh 
              .mail, .m, .bdf, .off, .p3d, .stl, .wrl, .vtk, .dat, .ply2, .su2, .neu, .x3d

        Args:
            filename (str): The filename
        """
        logger.trace(f'Writing geometry to {filename}')
        gmsh.write(filename)
        
    def set_solver(self, solver: EMSolver | Solver):
        """Set a given Solver class instance as the main solver. 
        Solvers will be checked on validity for the given problem.

        Args:
            solver (EMSolver | Solver): The solver objects
        """
        logger.trace(f'Setting solver to {solver}')
        self.mw.solveroutine.set_solver(solver)
        
    ############################################################
    #                     DEPRICATED FUNCTIONS                #
    ############################################################

    def define_geometry(self, *args):
        """DEPRICATED VERSION: Use .commit_geometry()
        """
        logger.warning('define_geometry() will be derpicated. Use commit_geometry() instead.')
        self.commit_geometry(*args)
        
class SimulationBeta(Simulation):
    
    
    def __post_init__(self):
        
        self.mesher.set_algorithm(Algorithm3D.HXT)
        logger.debug('Setting mesh algorithm to HXT')
        
        
    def _reset_mesh(self):
        #gmsh.clear()
        gmsh.model.mesh.clear()
        
        self.mw.reset(_reset_bc = False)
        self.state.reset_mesh()
        
    def adaptive_mesh_refinement(self, 
                                 max_steps: int = 6,
                                 min_refined_passes: int = 1,
                                 convergence: float = 0.02,
                                 magnitude_convergence: float = 2.0,
                                 phase_convergence: float = 180,
                                 refinement_ratio: float = 0.9,
                                 growth_rate: float = 3,
                                 minimum_refinement_percentage: float = 20.0, 
                                 error_field_inclusion_percentage: float = 5.0,
                                 frequency: float = None,
                                 show_mesh: bool = False) -> SimulationDataset:
        """ A beta-version of adaptive mesh refinement.

        Convergence Criteria:
            (1): max(abs(S[n]-S[n-1]))
            (2): max(abs(abs(S[n]) - abs(S[n-1])))
            (3): max(angle(S[n]/S[n-1])) * 180/π
        
        Args:
            max_steps (int, optional): The maximum number of refinement steps. Defaults to 6.
            min_refined_passes (int, optional): The minimum number of refined passes. Defaults to 1.
            convergence (float, optional): The S-paramerter convergence (1). Defaults to 0.02.
            magnitude_convergence (float, optional): The S-parameter magnitude convergence (2). Defaults to 2.0.
            phase_convergence (float, optional): The S-parameter Phase convergence (3). Defaults to 180.
            refinement_ratio (float, optional): The size reduction of mesh elements by original length. Defaults to 0.75.
            growth_rate (float, optional): The mesh size growth rate. Defaults to 3.0.
            minimum_refinement_percentage (float, optional): The minimum mesh size increase . Defaults to 15.0.
            error_field_inclusion_percentage (float, optional): A percentage of tet elements to be included for refinement. Defaults to 5.0.
            frequency (float, optional): The refinement frequency. Defaults to None.
            show_mesh (bool, optional): If the intermediate meshes should be shown (freezes simulation). Defaults to False

        Returns:
            SimulationDataset: _description_
        """
        from .physics.microwave.adaptive_mesh import select_refinement_indices, reduce_point_set, compute_convergence
        
        
        max_freq = np.max(self.mw.frequencies)
        
        if frequency is not None:
            max_freq = frequency
        
        S_matrices: list[np.ndarray] = []

        last_n_tets: int = self.mesh.n_tets
        logger.info(f'Initial mesh has {last_n_tets} tetrahedra')  
        
        passed = 0
        
        self.state.stash()
        
        for step in range(1,max_steps+1):
            
            self.data.sim.new(iter_step=step)
            
            data = self.mw._run_adaptive_mesh(step, max_freq)
            
            field = data.field[-1]
            
            Smat_new = data.scalar[-1].Sp
            S_matrices.append(Smat_new)
            
            if step > 1:
                S0 = S_matrices[-2]
                S1 = S_matrices[-1]
                conv_complex, conv_mag, conv_phase = compute_convergence(S0, S1)
                logger.info(f'Pass {step}: Convergence = {conv_complex:.3f}, Mag = {conv_mag:.3f}, Phase = {conv_phase:.1f} deg')
                if conv_complex <= convergence and conv_phase < phase_convergence and conv_mag < magnitude_convergence:
                    logger.info(f'Pass {step}: Mesh refinement passed!')
                    passed += 1
                else:
                    passed = 0
            
            if passed >= min_refined_passes:
                logger.info('Adaptive mesh refinement successfull')
                break
            
            error, lengths = field._solution_quality()
                
            idx = select_refinement_indices(error, error_field_inclusion_percentage/100)
            idx = idx[::-1]
            
            self.mesher.add_refinement_points(self.mw.mesh.centers[:,idx], lengths[idx])
            
            logger.debug(f'Pass {step}: Adding {len(idx)} new refinement points.')
                
            new_ids = reduce_point_set(self.mesher._amr_coords, growth_rate, self.mesher._amr_sizes, refinement_ratio, 0.20)
            
            logger.debug(f'Pass {step}: Removing {self.mesher._amr_coords.shape[1] - len(new_ids)} points from {self.mesher._amr_coords.shape[1]} to {len(new_ids)}')
            
            self.mesher._amr_coords = self.mesher._amr_coords[:,new_ids]
            self.mesher._amr_sizes = self.mesher._amr_sizes[new_ids]
            
            
            while True:
                
                self._reset_mesh()
                
                logger.debug(f'Pass {step}: Adding {len(idx)} refinement points.')
                
                self.mesher.set_refinement_function(refinement_ratio, growth_rate, 1.0)
                
                self.generate_mesh(True)
                
                percentage = (self.mesh.n_tets/last_n_tets - 1) * 100
                logger.info(f'Pass {step}: New mesh has {self.mesh.n_tets} (+{percentage:.1f}%) tetrahedra.')  
                
                if percentage < minimum_refinement_percentage:
                    logger.debug('Not enough mesh refinement, decreasing mesh size constraint.')
                    refinement_ratio = refinement_ratio * 0.9
                    logger.debug(f'New refinement ratio: {refinement_ratio}')
                    continue
                
                if percentage > 2*minimum_refinement_percentage and refinement_ratio < 0.99:
                    logger.debug('Too much mesh refinement, decreasing mesh size constraint.')
                    refinement_ratio = refinement_ratio ** (1-np.log(percentage/minimum_refinement_percentage)/4)
                    logger.debug(f'New refinement ratio: {refinement_ratio}')
                    

                last_n_tets = self.mesh.n_tets
                break
            
            if show_mesh:
                self.view(plot_mesh=True, volume_mesh=True)
        
        if passed < min_refined_passes:
            logger.warning('Adaptive mesh refinement did not converge!')
        
        old = self.state.reload()
        return old
    
    
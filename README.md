# HHGtoolkit
Libraries for plotting HHG and TDSE data from the HDF5 files and analysis.

Library developed by Tadeas Nemec.

## Contains
Library contains in total 6 submodules: `conversions.py`, `Hfn2.py`, `init_parameters.py`, `plotting.py`, `process.py`, `utilites.py`.

`conversions.py`: methods for unit conversion.

`Hfn2.py`: Hankel transform methods.

`init_parameters.py`: contains methods for easier data generation for the HHG code.

`plotting.py`: plotting methods including spectrum, fields, Gabor transform plotting etc.

`process.py`: some numerical methods and data processing methods.

`utilites.py`: Data and Dataset classes for HDF5 files loading and Beam class for gaussian beams plotting and parameters.

## Installation
1. Download the directory from Github: https://github.com/nemectad/HHGtoolkit.git
2. In terminal/CLI: move into the Download directory and into the root folder of the module.
3. Run the following command in your base environment:
  ```bash
  python3 setup.py install
  ```
4. All done.

Now you should by able to access the methods contained in the module simply by typing
```python
import HHGtoolkit
```
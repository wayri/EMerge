# ✅ Installing `scikit-umfpack` with Conda on Windows

This guide walks you through setting up **UMFPACK** (from SuiteSparse) to work with Python to use it for `emerge`.

**IMPORTANT**: Due to the assumption of windows use, all commands are using pip install commands using `python -m pip install ....`. Try what works for your system.

```bash
pip install ...
python -m pip install ...
py -m pip install ...
```

---

# The easy way
Install conda(miniconda) 

[Download Miniconda →](https://docs.conda.io/en/latest/miniconda.html)

And then do:
```bash
conda install conda-forge::scikit-umfpack
```

This should work, if not you can try the hard way!

# The hard way

## Step 0: (Optional) Uninstall conflicting packages first

Before you begin, it’s safest to uninstall any conflicting or prebuilt packages like `emerge`, `numpy`, or `scipy`. You’ll reinstall them later in step 6.

```bash
python -m pip uninstall emerge
python -m pip uninstall numpy
python -m pip uninstall scipy
python -m pip uninstall numba
python -m pip uninstall numba-progress
```

---

## Step 1: Install Miniconda (or Anaconda)

[Download Miniconda →](https://docs.conda.io/en/latest/miniconda.html)

---

## Step 2: Create a Conda environment (Python 3.10 or 3.11 recommended)
The latest version of scikit-umfpack is not compatible with python 3.13 so this likely only works on lower python versions. Its tried on 3.10.

```bash
conda create -n umf-env python=3.10 -y
conda activate umf-env
```

or switch to 3.10 in your main conda environment
```bash
conda install -y python=3.10
```

---

## Step 3: Install required packages

```bash
conda install anaconda::suitesparse
conda install meson swig
conda install -c conda-forge compilers
conda install -c conda-forge m2w64-toolchain
conda install -c conda-forge openblas
python -m pip install meson-python
```
**Extra**
If you deinstalled numpy and scipy, install them first:
```bash
python -m pip install numpy
python -m pip install scipy
```
---

## Step 4: Create a `nativefile.ini` file

In the directory where you will run the build command, create a file called:

```
nativefile.ini
```

Paste the following into it:

```ini
[binaries]
c = 'x86_64-w64-mingw32-gcc'
cpp = 'x86_64-w64-mingw32-g++'

[properties]
umfpack-libdir = '''C:/Path/To/miniconda3/Library/lib'''
umfpack-includedir = '''C:/Path/To/miniconda3/Library/include/suitesparse'''
```

> Replace `C:/Path/To` with your actual Miniconda installation path.

---

## Step 5: Install `scikit-umfpack` from source

Run this command **from the same folder where `nativefile.ini` is**:

Bash:
```bash
python -m pip install scikit-umfpack --no-build-isolation -Csetup-args="--native-file=$(pwd)/nativefile.ini"
```
Powershell
```powershell
python -m pip install scikit-umfpack --no-build-isolation -Csetup-args="--native-file=$((Get-Location).Path)/nativefile.ini"
```

---

## Step 6 (optional): Reinstall the packages you uninstalled earlier

```bash
python -m pip install numba
python -m pip install numba-progress

# If you're using the emerge FEM project
python -m pip install --no-deps emerge
```
> I am working on configuring the dependencies of emerge better so that this step should not be necessary.
---

## Done!

You should now have `scikit-umfpack` installed and working with SuiteSparse in your Conda environment.
Test it with:

```python
import scikits.umfpack
print("UMFPACK loaded successfully!")
```

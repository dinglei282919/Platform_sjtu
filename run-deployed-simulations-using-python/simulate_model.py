"""
Utility function to locate, import and initialize simulate_model package.

@author: Murali Yeddanapudi
@date: Nov-2025
"""


def _prepare_mcr_path(mcr_root):
    if not mcr_root:
        return None

    import os
    import platform

    root = os.path.abspath(os.fspath(mcr_root))
    system = platform.system()
    if system == "Windows":
        arch = "win64"
        path_var = "PATH"
        runtime_file = "mclmcrrt24_2.dll"
    elif system == "Linux":
        arch = "glnxa64"
        path_var = "LD_LIBRARY_PATH"
        runtime_file = "libmwmclmcrrt.so.24.2"
    elif system == "Darwin":
        arch = "maca64" if platform.mac_ver()[-1] == "arm64" else "maci64"
        path_var = "DYLD_LIBRARY_PATH"
        runtime_file = "libmwmclmcrrt.24.2.dylib"
    else:
        raise RuntimeError(f"Unsupported platform: {system}")

    runtime_dir = os.path.join(root, "runtime", arch)
    bin_dir = os.path.join(root, "bin", arch)
    extern_dir = os.path.join(root, "extern", "bin", arch)
    runtime_library = os.path.join(runtime_dir, runtime_file)
    if not os.path.exists(runtime_library):
        raise FileNotFoundError(f"MATLAB Runtime library was not found: {runtime_library}")

    current = os.environ.get(path_var, "")
    current_parts = current.split(os.pathsep) if current else []
    current_parts_lower = {part.lower() for part in current_parts}
    prepend_parts = [
        part
        for part in (runtime_dir, bin_dir, extern_dir)
        if os.path.isdir(part) and part.lower() not in current_parts_lower
    ]
    if prepend_parts:
        os.environ[path_var] = os.pathsep.join(prepend_parts + current_parts)
    return runtime_library


def load_and_init_pkg(modelName: str, base_dir=None, mcr_root=None):
    """
    Utility function to locate, import and initialize the package to simulate
    the given deployed Simulink model.

    It is assumed that the package is named "simulate_<modelName>" and is located
    under the "./simulate_<modelName>/(pcwin64|glnxa64)" directory.

    The included MATLAB function build_python_package_to_run_deployed_model.m can
    be used to create the package for a given deployed Simulink model.

    Args:
        modelName (str): Name of the deployed Simulink model
        base_dir: Directory containing simulate_<modelName>.
        mcr_root: Optional MATLAB Runtime/MATLAB root used to prepare runtime paths.
    Returns:
        mdl: Initialized simulate_model package object
    """

    pkgName = "simulate_"+modelName
    print(f"Locating '{pkgName}' package for model '{modelName}' ...")

    import os
    import sys
    import platform
    _prepare_mcr_path(mcr_root)
    baseDir = os.path.abspath(os.fspath(base_dir)) if base_dir else os.path.dirname(os.path.abspath(__file__))
    plat = platform.system()
    if plat == "Windows":
        pkgDir = os.path.join(baseDir,pkgName,'pcwin64')
    elif plat == "Linux":
        pkgDir = os.path.join(baseDir,pkgName,'glnxa64')
    else:
        raise RuntimeError(f"Unsupported platform: {plat}")

    # Walk under the given root (e.g. "simulate_model1")
    for dirpath, dirnames, filenames in os.walk(pkgDir):
        if "__init__.py" in filenames and os.path.basename(dirpath) == pkgName:
            # dirpath is .../dist-packages/simulate_model1
            parentDir = os.path.dirname(dirpath)  # .../dist-packages
            sys.path.insert(0, parentDir)
            print("Importing "+pkgName+" package from "+parentDir)
            import importlib
            simMdl = importlib.import_module(pkgName)
            print("Initializing "+pkgName+" ...")
            mdl = simMdl.initialize()
            return mdl
    raise FileNotFoundError(f"No package '{pkgName}' for model '{modelName}' under {pkgDir}")
###

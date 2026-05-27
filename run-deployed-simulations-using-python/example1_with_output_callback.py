"""
This example shows how to simulate a deployed Simulink model with a callback
function that is called periodically during simulation.

Note that output callback function (outputFcn defined below) is executed in a
separate Python process started by the MATLAB Runtime that runs the deployed
Simulink model simulation initiated from main(). So outputFcn cannot access
variables that are initialized in main()

The deployed simulink model is expected to be packaged with the included MATLAB
function simulate.m into a Python package named simulate_<modelName>. You can
use the included MATLAB function build_python_package_to_run_deployed_model.m
to create the python package.

It is assumed that the MATLAB Runtime is installed and available on the system. See:
    1. https://www.mathworks.com/help/compiler/install-the-matlab-runtime.html
    2. https://www.mathworks.com/help/compiler/mcr-path-settings-for-run-time-deployment.html
for instructions to download, install and configure the MATLAB Runtime.
    
@author: Murali Yeddanapudi
@date: Nov-2025
"""

def outputFcn(simTime, results):
    """
    Callback function called periodically during simulation. Simply prints the
    current simulation time and number of  logged data points.

    Note that this function is executed in a separate Python process started
    by the MATLAB Runtime that runs the deployed Simulink model simulation
    initiated from main(). So this function cannot access variables that are
    initialized in main()

    Args:
        simTime (float): Current simulation time
        results (dict): Dictionary containing logged signal
        data since last call to this function.
    
    Returns:
        stopReq (bool): True to request simulation stop, False to continue
    """

    # Names 'x1', 'x2', etc., correspond to the logged signals in the model
    import numpy as np
    tx1 = np.array(results['x1']['Time']).flatten()
    dx1 = np.array(results['x1']['Data']).flatten()
    tx2 = np.array(results['x2']['Time']).flatten()
    dx2 = np.array(results['x2']['Data']).flatten()

    print("@T = ", simTime, "; #x1 = ", tx1.size, "; #x2 = ", tx2.size)
    # request simulation if simTime exceeds 100
    stopReq = simTime > 100
    if stopReq:
       print("@T = ", simTime, "; Requesting simulation stop")                
    return stopReq
###

def main(progName: str, modelName: str = None):
    """
    Main function to simulate the given deployed Simulink model with a callback
    function that is called periodically during simulation.
    
    Args:
        progName (str): Name of the current program
        modelName (str): Name of the deployed Simulink model
    """

    # import and initialize simulate_<modelName> package
    import simulate_model
    mdl = simulate_model.load_and_init_pkg(modelName)

    # Tell MATLAB how to call the outputFcn defined above
    # Note that this function is called from a separate Python process started
    # by the MATLAB Runtime that runs the deployed Simulink model simulation.
    mlOutputFcn = "py." + progName + ".outputFcn"
    
    print("Calling simulate ...")
    tuneableParams = {
        'dx2min': -3.0, # Specify a new value for dx2min
        'dx2max':  4.0  # Specify a new value for dx2max
    }
    res0 = mdl.simulate(
        'ModelName', modelName,
        'StopTime', 1000,
        'TunableParameters', tuneableParams,
        'OutputFcn', mlOutputFcn,
        'OutputFcnDecimation', 10)

    input("Press enter to exit ...")
    print("Terminating ...")
    mdl.terminate() # stop the MATLAB Runtime
###

if __name__ == '__main__':
    import os
    import argparse
    progName = os.path.splitext(os.path.basename(__file__))[0]
    parser = argparse.ArgumentParser(
        prog=progName,
        description="python "+progName+".py --mdl modelName (default: model1)")
    parser.add_argument(
        '-m', '--mdl',
        dest='modelName',
        default="model1",
        help='Name of the deployed Simulink model')
    args = parser.parse_args()
    main(progName=progName, modelName=args.modelName)

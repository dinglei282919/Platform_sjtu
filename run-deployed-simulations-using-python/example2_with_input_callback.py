"""
This example shows how to simulate a deployed Simulink model with a callback
function that is called during simulation when new input signal values are needed.

The input function is required to return the input signal value at the simTime
specified, but it can also return input signal values at future time points.
This will reduce the number of calls to the input function during simulation
and improve performance.

The simulation results logged since the last call to the input function are
passed in to the input function, which uses this data to update a scrolling plot
of the logged signals. The new input signal values are also plotted in the same
scrolling plot.

The input function has a second return value to request simulation stop.

Note that input callback function (inputFcn defined below) is executed in a
separate Python process started by the MATLAB Runtime that runs the deployed
Simulink model simulation initiated from main(). So inputFcn cannot access
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

def inputFcn(portNumber, simTime, resultsSoFar):
    """
    Callback function called during simulation when new input signal values
    are needed.

    Note that this function is executed in a separate Python process started
    by the MATLAB Runtime that runs the deployed Simulink model simulation
    initiated from main(). So this function cannot access variables that are
    initialized in main()

    Args:
        portNumber (int): Input port number (1-based)
        simTime (float): Current simulation time
        resultsSoFar (dict): Dictionary containing logged signal
        data since the start of simulation.

    Returns:
        u (float array): (Nx1) of input signal values at time points
        tu = [simTime, simTime+1, ..., simTime+(N-1)]. N is variable
        stopReq (bool): True to request simulation stop, False to continue
    """

    import numpy as np
    if resultsSoFar:
        t1 = np.array(resultsSoFar['x1']['Time']).flatten()
        x1 = np.array(resultsSoFar['x1']['Data']).flatten()
        t2 = np.array(resultsSoFar['x2']['Time']).flatten()
        x2 = np.array(resultsSoFar['x2']['Data']).flatten()
        print("@T = ", simTime, ": Received ", t1.size, " x1 points, ", t2.size, " x2 points")
    else:
        # first time
        t1 = np.array([])
        x1 = np.array([])
        t2 = np.array([])
        x2 = np.array([])
        inputFcn.rng = np.random.default_rng(0) # for repeatability

    stopReq = simTime > 500
    if stopReq:
        nU = 1
        u = float('nan') 
        tu = simTime
    else:
        # return a variable number of input signal values at time points tu.
        # Note that tu is an array that starts at simTime and steps forward
        # by 1 second (=uST the sample time of input port in the model).
        # We only return u array because tu is fixed by the sample time.
        # number of input signal values can be varied to control how often
        # this function is called during simulation.
        nU = inputFcn.rng.integers(5, 25) 
        u = inputFcn.rng.uniform(-0.5, 0.5, nU)    
        tu = simTime + np.arange(nU, dtype=float)

    # plot x1 and x2 from the resultsSoFar and input u at times tu into the future
    import scrolling_plot
    scrolling_plot.update(
        [('x1',t1,x1),
         ('x2',t2,x2),
         ('u',tu,u,'step')],
        timeSpan=50)

    if stopReq:
        print("@T = ", simTime, ": Requesting simulation stop")
    else:
        print("@T = ", simTime, ": Returning ", nU, " input values")
    return u, stopReq

def main(progName: str, modelName: str = None):
    """
    Main function to simulate the given deployed Simulink model with
    input callback function defined above.
    
    Args:
        progName (str): Name of the current program
        modelName (str): Name of the deployed Simulink model
    """

    # import and initialize simulate_<modelName> package
    import simulate_model
    mdl = simulate_model.load_and_init_pkg(modelName)

    # Tell MATLAB how to call the inputFcn defined above
    # Note that this function is called from a separate Python process started
    # by the MATLAB Runtime that runs the deployed Simulink model simulation.
    mlInputFcn = "py." + progName + ".inputFcn"
    
    print("Calling simulate ...")
    results = mdl.simulate(
        'ModelName', modelName,
        'StopTime', 1000,
        'InputFcn', mlInputFcn)

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

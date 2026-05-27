"""
This example shows how to simulate a deployed Simulink model with different
parameter and external input signal values using the MATLAB Runtime.

The model is simulated four times with different parameter and external input
signal values, and the simulation results are retrieved and plotted after
each simulation.

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

def main(modelName: str = None):
    """
    Main function to simulate the given deployed Simulink model multiple times
    with different parameter and external input signal values, and plot the
    simulation results.

    Args:
        modelName (str): Name of the deployed Simulink model
    """

    # Setup the figure to plot simulation results
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_ylim([-4, 3])
    ax.set_title("Plot of x1 from four simulations")
    plt.pause(0.1)  # allow GUI backends to refresh
    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def update_plot(res, col, lbl):
        """
        Utility function to plot simulation result and update the legend
        """
        # Names 'x1', 'x2', etc., correspond to the logged signals in the model
        ax.plot(res['x1']['Time'], res['x1']['Data'],
                color=col, linewidth=2, label=lbl)
        ax.legend(loc='lower right')
        plt.pause(0.1)  # allow GUI backends to refresh
    ###

    # import and initialize simulate_<modelName> package
    import simulate_model
    mdl = simulate_model.load_and_init_pkg(modelName)

    print("Running 1st sim with default parameter values")
    res0 = mdl.simulate('ModelName', modelName)
    update_plot(res0, cols[0], 'x1 from 1st sim with default setting')

    print("Running 2nd sim with new values for dx2min and dx2max parameters")
    tunableParams = {
        'dx2min': -3.0, # Specify a new value for dx2min
        'dx2max':  4.0  # Specify a new value for dx2max
        }
    res1 = mdl.simulate(
        'ModelName', modelName,
        'TunableParameters',tunableParams)
    update_plot(res1, cols[1], 'x1 from 2nd sim with limits on dx2')

    print("Running 3rd sim with a non zero input signal")
    # Note that in the model the input u is sampled at a fixed time interval
    # uST (=1s), so the time axis for the input values is not specified, it
    # is implicitly assumed that the time values are uniformly spaced at 1s
    # (=uST) interval.
    # u(t) = 2 for t in [1,2), -2 for t in [6,8), 0 otherwise
    import numpy as np
    u = np.concatenate([np.zeros(1), 2*np.ones(1), np.zeros(3), -2*np.ones(2), np.zeros(1)])
    res2 = mdl.simulate(
        'ModelName', modelName,        
        'ExternalInput',u)
    # plot u using step, since it is a "discrete" (i.e., not continuous)
    ax.step(res2['u']['Time'], res2['u']['Data'],
            color=cols[2], linewidth=2,
            label='input u in 3rd and 4th sims')
    update_plot(res2, cols[3], 'x1 from 3rd sim with input u')

    print("Running 4th sim with dx2min, dx2max and non-zero input signal")
    tunableParams = {
        'dx2min': -3.0, # Specify a new value for dx2min
        'dx2max':  4.0  # Specify a new value for dx2max
        }
    u = np.concatenate([np.zeros(1), 2*np.ones(1), np.zeros(3), -2*np.ones(2), np.zeros(1)])
    res3 = mdl.simulate(
        'ModelName', modelName,
        'TunableParameters',tunableParams,
        'ExternalInput',u)
    update_plot(res3, cols[4], 'x1 from 4th sim with limits on dx2 and input u')

    print("Close the plot window to exit ...")
    plt.show()
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
    main(modelName=args.modelName)

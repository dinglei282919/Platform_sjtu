function res = simulate(args)
% Utility function to configure and simulate a Simulink model in deployment
% mode with the specified parameter and input signal values.
% 
% Inputs:
%    ModelName:
%       Name of the model to simulate
%
%    StopTime:
%       Simulation stop time, default value (nan) uses the stop time in
%       the model
%
%    TunableParameters
%       A struct where the fields are the tunable referenced workspace
%       variables with the values to use for the simulation.
%
%    ExternalInput:
%       External Input signal, default is empty.
%
%    InputFcn:
%       Specify a callback function that will be called when new batch of
%       input signal values are needed during simulation. Expected syntax:
%           [newValues,stopRequested] = inputFcn(portNumber, simTime, resultsSinceLastCall)
%       Input port is expected to have a discrete periodic sample time and
%       scalar valued.
%       It is an error to specify both ExternalInput and InputFcn.
%
%    OutputFcn:
%       Specify a callback function that will be called during simulation
%       with the simulation results collected this far. Expected syntax:
%          stopRequested = outputFcn(simTime, resultsSinceLastCall)
%
%    OutputFcnDecimation:
%       Specify how often (in terms of number of time steps) to call the
%       OutputFcn, default is one.
%
%    Debug:
%       If true, display debug information during simulation. Default is false.
%
%    Values of nan or empty for parameters and inputs indicate that sim
%    should run with the default values specified in the model
% 
% Outputs:
%    res: A structure with the time and data values of the logged signals.

% By: Murali Yeddanapudi, Nov-2025

arguments
    args.ModelName {mustBeTextScalar}
    args.StopTime (1,1) double = nan
    args.TunableParameters = []
    args.ExternalInput (1,:) {mustBeNumericOrLogical} = []
    args.InputFcn {mustBeFunctionHandle} = @emptyFunction
    args.OutputFcn  {mustBeFunctionHandle} = @emptyFunction
    args.OutputFcnDecimation (1,1) {mustBeInteger, mustBePositive} = 1
    args.Debug (1,1) logical = false
end

    %% Create the SimulationInput configured for deployment
    si = Simulink.SimulationInput(args.ModelName);
    si = simulink.compiler.configureForDeployment(si);
    
    %% Load the StopTime into the SimulationInput object
    if ~isnan(args.StopTime)
        si = si.setModelParameter('StopTime', num2str(args.StopTime));
    end
    
    %% Load the specified tunable parameters into the simulation input object
    if isstruct(args.TunableParameters) 
        tpNames = fieldnames(args.TunableParameters);
        for itp = 1:numel(tpNames)
            tpn = tpNames{itp};
            tpv = args.TunableParameters.(tpn);
            si = si.setVariable(tpn, tpv);
        end
    end

    if ~isempty(args.ExternalInput) && ~isequal(args.InputFcn, @emptyFunction)
        error("Specify either ExternalInput or InputFcn, not both");
    end

    %% Load the external input into the SimulationInput object
    if ~isempty(args.ExternalInput)
        % External input is expected to have discrete (i.e., not continuous)
        % sample time. Hence the time points where it is sampled are
        % pre-determined in the model and hence only specify the data values
        % here using the struct with empty time field as described in Guy's
        % blog:
        % https://blogs.mathworks.com/simulink/2012/02/09/using-discrete-data-as-an-input-to-your-simulink-model/
        uStruct.time = [];
        uStruct.signals.dimensions = 1;
        % values needs to be column vector
        uStruct.signals.values = reshape(args.ExternalInput,numel(args.ExternalInput),1);
        si.ExternalInput = uStruct;
    end
    
    %% InputFcn
    simTimeAtLastExtInpFcnCall = -inf; % invalid
    function u = locInputFcn(pn, simTime)
        % Get new results that have been logged since the last call
        locRes = []; % assume
        if (isfinite(simTimeAtLastExtInpFcnCall) && ...
            simTimeAtLastExtInpFcnCall < simTime)
            locSO = simulink.compiler.getSimulationOutput(args.ModelName);
            locRes = extractResults(locSO, simTimeAtLastExtInpFcnCall);
        end
        if args.Debug
            fprintf("Calling InputFcn at simTime=%f\n", simTime);
        end        
        uAndStopReq = feval(args.InputFcn, pn, simTime, locRes);
        u = double(uAndStopReq{1});
        stopReq = uAndStopReq{2};
        if args.Debug
            fprintf("InputFcn returned %d new input values and stopReq=%d\n", numel(u), stopReq);
        end
        if stopReq
            simulink.compiler.stopSimulation(args.ModelName);
        end
        simTimeAtLastExtInpFcnCall = simTime;
    end
    
    if ~isequal(args.InputFcn, @emptyFunction)
        % TODO: Check that root input ports have discrete sample time
        % Calling Python functions from deployed code requires out-of-process execution
        if startsWith(args.InputFcn, "py.") && isdeployed()
            pyenv('ExecutionMode', 'OutOfProcess');
        end
        si = simulink.compiler.setExternalInputsFcn(si, @(pn,t) locInputFcn(pn, t));
    end
    
    %% OutputFcn
    simTimeAtLastPostStepCall = nan;
    function locPostStepFcn(simTime)
        locSO = simulink.compiler.getSimulationOutput(args.ModelName);
        locRes = extractResults(locSO, simTimeAtLastPostStepCall);
        stopRequested = feval(args.OutputFcn, simTime, locRes);
        if stopRequested
            simulink.compiler.stopSimulation(args.ModelName);
        end
        simTimeAtLastPostStepCall = simTime;
    end
    if ~isequal(args.OutputFcn, @emptyFunction)
        % Calling Python functions from deployed code requires out-of-process execution
        if startsWith(args.OutputFcn, "py.") && isdeployed()
            pyenv('ExecutionMode', 'OutOfProcess');
        end
        si = simulink.compiler.setPostStepFcn(si, @locPostStepFcn, ...
            'Decimation', args.OutputFcnDecimation);
    end
    
    %% call sim
    so = sim(si);
    
    %% Extract the simulation results
    % Package the time and data values of the logged signals into a structure
    res = extractResults(so,nan);

end % simulate

function res = extractResults(so, prevSimTime)
    % Package the time and data values of the logged signals into a structure
    ts = simulink.compiler.internal.extractTimeseriesFromDataset(so.logsout);
    for its=1:numel(ts)
        if isfinite(prevSimTime)
            idx = find(ts{its}.Time > prevSimTime);
            res.(ts{its}.Name).Time = ts{its}.Time(idx);
            res.(ts{its}.Name).Data = ts{its}.Data(idx);
        else
            res.(ts{its}.Name).Time = ts{its}.Time;
            res.(ts{its}.Name).Data = ts{its}.Data;
        end
    end
end

function mustBeFunctionHandle(fh)
    if ~isa(fh,'function_handle') && ~ischar(fh) && ~isstring(fh)
        throwAsCaller(error("Must be a function handle"));
    end
end

function emptyFunction
end

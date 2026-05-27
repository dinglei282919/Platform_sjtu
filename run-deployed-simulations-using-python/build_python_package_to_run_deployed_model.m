function build_python_package_to_run_deployed_model(args)
% Utility to build and install a python package to simulate a deployed
% Simulink model from python using MATLAB Runtime
%
% Inputs:
%    ModelName:
%       Name of the model to deploy.
%       The python package will be named simulate_<ModelName>
%
% The following artifacts will be created:
%    1. The installer for the python package will created in:
%       ./simulate_<ModeName>_installed/<computer>
%    2. The python package will be installed under
%       ./simulate_<ModeName>/<computer>
%
% The following products are required:
%    MATLAB
%    MATLAB Compiler
%    MATLAB Compiler SDK
%    Simulink
%    Simulink Compiler

% By Murali Yeddanapudi, Nov-2025

arguments
    args.ModelName {mustBeTextScalar}
end

    %% Setup: create and cd to a tempdir, will be cleaned up on exit
    plat = lower(computer);
    origDir = pwd;
    addpath(origDir);
    tempDir = tempname;
    mkdir(tempDir);
    cd(tempDir);
    cleanup = onCleanup(@()locOnCleanup(origDir,tempDir));
    fprintf('\n### Created and cd''ed to %s\n', pwd);

    %% Make sure the model simulates in deployment mode
    fprintf('\n### Test simulating the model in deployment mode ..\n');
    simulate('ModelName',args.ModelName);

    %% Setup to build the python package installer
    appFile = which('simulate');
    pkgName = "simulate_"+args.ModelName;
    outDir = fullfile(origDir,pkgName+"_installer",plat);
    if exist(outDir,'dir')
        rmdir(outDir,'s');
    end

    %% Build the python package
    fprintf("\n### Building python package '%s' in %s\n", pkgName, outDir);
    compiler.build.pythonPackage( ...
        appFile, ...
        'OutputDir',outDir, ...
        'PackageName',pkgName, ...
        'AdditionalFiles',which(args.ModelName));

    %% Ask if the generated python package should be installed
    yesno = input("==> Install the Python package? Y/N [Y]: ","s");
    if isempty(yesno), yesno = 'y'; end
    yesno = upper(yesno);
    pkgDir = fullfile(origDir,pkgName,plat);

    %% locate python
    python = 'python';
    [stat,~] = system(python+" --version");
    if stat
        python = 'python3';
        [stat,msg] = system(python+" --version");
        if stat
            warning(msg);
            yesno = 'N';
        end
    end

    %% Install the generated python package
    installCmd = python+" setup.py install --prefix="+'"'+pkgDir+'"';    
    if isequal(yesno,'Y')
        fprintf('\n### Running %s\n', installCmd);
        cd(outDir);
        system(installCmd);
        fprintf('\n### Installed %s\n', pkgDir);
    else
        fprintf('\n### Run the commands below to install the python package\n');
        fprintf('    cd %s\n', outDir);
        fprintf('    %s\n', installCmd);
    end

    %% Display usage instructions
    [mcrinstallerPath, ~, ~, ~] = mcrinstaller;
    fprintf('\n### To use the installed python package, your python code needs MATLAB Runtime\n');
    fprintf('    1. Copy and unzip the MATLAB Runtime installer from here:\n');
    fprintf('          %s\n',mcrinstallerPath);
    fprintf('    2. Instructions to install (and also download, if you have not copied from above) are here:\n');
    fprintf('          https://www.mathworks.com/help/compiler/install-the-matlab-runtime.html\n');
    fprintf('    3. Instructions for your python code to find the installed MATLAB Runtime are here:\n');
    fprintf('          https://www.mathworks.com/help/compiler/mcr-path-settings-for-run-time-deployment.html\n');
    fprintf('\n');
end

function locOnCleanup(origDir, tempDir)
    cd(origDir);
    rmdir(tempDir,'s');
    rmpath(origDir);
end

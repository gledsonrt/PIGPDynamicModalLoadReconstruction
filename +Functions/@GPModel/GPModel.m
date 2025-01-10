classdef GPModel < handle
    % Handles the physics-informed GP model for dynamic force reconstruction
    properties
        M, C, K, modeList
        kernels
        trainingData
        hyps0, hypsOpt
        alpha, kinv
        fVec, sVec
        jitter = 1e-15;
        freqFilter = [1,1.96];
        freqMin = 0;
        freqMax = Inf;
        woodburyFlag = true;
        analytDerivs = false;
        parallelWorkers = 0;
        verbose        
    end
    methods
        % Constructor
        function self = GPModel(M, C, K, varargin)
            % Parse inputs
            p = inputParser();
            p.CaseSensitive = false;
            p.addOptional('frequencies', []);
            p.addOptional('parallelWorkers', 0);
            p.addOptional('woodburyFlag', true);
            p.addOptional('analytDerivs', false);
            p.addOptional('verbose', 1);
            p.addOptional('freqFilter', [0, 0]);
            p.addOptional('freqMin', 0);
            p.addOptional('freqMax', Inf);
            p.addOptional('jitter', 1e-15);
            p.parse(varargin{:});
            
            % Store system properties
            self.M = M;
            self.C = C;
            self.K = K;
            
            % Get inputs
            self.parallelWorkers = p.Results.parallelWorkers;
            self.woodburyFlag = p.Results.woodburyFlag;
            self.analytDerivs = p.Results.analytDerivs;
            self.jitter = p.Results.jitter;
            self.verbose = p.Results.verbose;
            self.freqFilter = p.Results.freqFilter;
            self.freqMin = p.Results.freqMin;
            self.freqMax = p.Results.freqMax;
            
            % Verbose options
            % 0 - no outputs at all
            % 1 - basic outputs (model creation, current training mode, current mode predition)
            % 2 - additional info on model training data
            % 3 - additional info of optimization steps
            if self.verbose > 0
                fprintf('Creating GP model\n');
            end
            
            % Store inputs
            self.fVec = p.Results.frequencies(:);        
            
            % Start parallel pool
            if self.parallelWorkers > 1
                p = gcp('nocreate');
                if ~isempty(p)
                    % delete(p)
                else
                    try
                        parpool(self.parallelWorkers);
                    catch
                        parpool('local');
                    end
                end
            end
        end
        
        % Declare additional functions
        addTrainingData(self, t, y, type, mode, noiseStdev)          
        [phi] = generateBasisFunction(self, t, f, spec, type)
        [NLML, dNLML] = likelihood(self, params, modeNumber)
        train(self, flagTrain)
        reset(self)
        tIdx = preprocess(self, varargin)
        [MuF,SigF,Results] = predict(self, t)  
    end
end
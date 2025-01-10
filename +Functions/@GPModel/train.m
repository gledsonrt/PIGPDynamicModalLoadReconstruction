function train(self, flagTrain)
    % Wrapper function for GP training, currently using NLML
    
    % Training is ON by default
    if ~exist('flagTrain', 'var'); flagTrain = true; end
    
    % Check available training data
    Ntraining = length(self.trainingData);
    self.modeList = unique([self.trainingData.mode]);
    Nmodes = length(self.modeList);
    
    % Prior for the spectrum: combine available info from training data
    for i = 1:Nmodes
        thisSpec = find(self.modeList(i) == [self.trainingData.mode]);
        freqVec = [];
        specLim = [];
        idxVec = [];
        for j = 1:length(thisSpec)   
            thisSpecLim = self.freqFilter(1)*mean(self.trainingData(thisSpec(j)).sVec) + self.freqFilter(2)*std(self.trainingData(thisSpec(j)).sVec);
            idx = (abs(self.trainingData(thisSpec(j)).sVec) > thisSpecLim & ...
                self.trainingData(thisSpec(j)).fVec >= self.freqMin & self.trainingData(thisSpec(j)).fVec <= self.freqMax); 
            freqVec = cat(1, freqVec, self.trainingData(thisSpec(j)).fVec(idx));
            idxVec = cat(1, idxVec, find(idx));
            specLim = cat(1, specLim, thisSpecLim);
        end
        freqVec = unique(freqVec);
        idxVec = unique(idxVec);
        for j = 1:length(thisSpec) 
            self.trainingData(thisSpec(j)).sVec = self.trainingData(thisSpec(j)).sVec(idxVec);
            self.trainingData(thisSpec(j)).fVec = freqVec;
        end
    end
    if isempty(self.fVec); self.fVec = freqVec; end
    
    % Check for speeding up using Woodbury matrix identity
    if self.woodburyFlag
        self.woodburyFlag = true(Nmodes, 1);
        for i = 1:Nmodes
            thisModeIndexes = find(self.modeList(i) == [self.trainingData.mode]);
            fLen = 2*length(self.trainingData(thisModeIndexes(1)).fVec);
            tLen = 0;
            for j = thisModeIndexes
                tLen = tLen + length(self.trainingData(j).t);
            end
            
            if tLen < fLen; self.woodburyFlag(i) = false; end
            if self.verbose > 0
                fprintf('Woodbury matrix identity flag for mode %02d set to %s\n', self.modeList(i), mat2str(self.woodburyFlag(i)));
            end
        end
    end

    % Start kernel cells
    self.kernels = cell(Nmodes,1);
    if self.verbose > 0; fprintf('Starting kernel building\n'); end
    for k = 1:Nmodes
        % List of mode indexes
        trainindModList = find(self.modeList(k) == [self.trainingData.mode]);
        
        % Verbose
        if self.verbose > 1
            fprintf('Generating kernels for mode %1.0f\n', self.modeList(k));
        end
        
        % Build training kernels
        modalBasisA = [];
        modalBasisB = [];
        for i = 1:length(trainindModList)
            thisPhi = self.generateBasisFunction(self.trainingData(trainindModList(i)).t, self.trainingData(trainindModList(i)).fVec, ...
                                            self.trainingData(trainindModList(i)).sVec, self.trainingData(trainindModList(i)).type);
            modalBasisA = cat(1, modalBasisA, thisPhi);
            modalBasisB = cat(2, modalBasisB, thisPhi');
        end        

        % Store basis functions and create kernel
        self.kernels{k}{1,1} = modalBasisA;
        self.kernels{k}{1,2} = modalBasisB;
        self.kernels{k}{1,3} = modalBasisA*modalBasisB;
    end
    
    % Set up initial parameters
    self.hyps0 = cell(Nmodes, 1);
    for j = 1:Nmodes
        self.hyps0{j} = 1;
        for i = 1:Ntraining
            if self.trainingData(i).mode == self.modeList(j)
                self.hyps0{j} = cat(2, self.hyps0{j}, self.trainingData(i).noise);
            end
        end
        self.hyps0{j} = log(self.hyps0{j});
    end

    % Optimization options
    if ~flagTrain; return; end
    if self.verbose > 0; fprintf('Starting parameter optimisation\n'); end
    if self.verbose > 2; verb = 'iter'; else; verb = 'none'; end
    if self.parallelWorkers > 1
        hypsOpt = cell(Nmodes, 1);
        alpha = cell(Nmodes, 1);
        kinv = cell(Nmodes, 1);
        hyps0 = self.hyps0;
        derivFlag = self.analytDerivs;
        woodbFlag = self.woodburyFlag;
        modeList = self.modeList;
        optFnc = @(x,m) self.likelihood(x, m);
        verbValue = self.verbose;
        parfor i = 1:Nmodes
            if verbValue > 1
                fprintf('Optimising parameters for mode %1.0f\n', modeList(i));
            end
            thisFunc = @(x) optFnc(x, i);
            if derivFlag && ~woodbFlag(i)
                options = optimoptions('fminunc', 'Display', verb, 'Algorithm', 'trust-region', 'SpecifyObjectiveGradient', true);
            else
                options = optimoptions('fminunc', 'Display', verb);
            end
            hypsOpt{i} = fminunc(thisFunc, hyps0{i}, options);
            alpha{i} = self.alpha{i};
            kinv{i} = self.kinv{i};
        end
        self.hypsOpt = hypsOpt;
        self.alpha = alpha;
        self.kinv = kinv;
    else
        self.hypsOpt = cell(Nmodes, 1);
        for i = 1:Nmodes
            if self.verbose > 1
                fprintf('Optimising parameters for mode %1.0f\n', self.modeList(i))
            end
            optFnc = @(x) self.likelihood(x, i);
            if self.analytDerivs && ~self.woodburyFlag
                options = optimoptions('fminunc', 'Display', verb, 'Algorithm', 'trust-region', 'SpecifyObjectiveGradient', true);
            else
                options = optimoptions('fminunc', 'Display', verb);
            end
            self.hypsOpt{i} = fminunc(optFnc, self.hyps0{i}, options);
        end
    end

end
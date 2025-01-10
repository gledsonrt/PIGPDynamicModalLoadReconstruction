function addTrainingData(self, t, y, type, mode, noiseStdev)
    % Adds training data for the GPModel
    % t             [N,1] vector of time-steps
    % y             [N,1] vector of responses (disp/vel/accel)
    % type          [1x1] type of response (1, 2, 3 / u, v, a / disp, vel, accel)
    % mode          [1x1] mode number
    % noiseStdev    [1x1] optional - initial assumption of noise in dataset
    
    % Data shape
    assert(any(size(t) == 1), 'Time data should be a vector.')
    assert(any(size(y) == 1), 'Measurement data should be a vector.')
    t = t(:);   y = y(:);
    assert(length(t) == length(y), 'Lengths of time and measurement vectors are different.');

    % Types
    switch type
        case {1, 'u', 'displacement'}
            type = 'u'; typeFull = 'displacement';
        case {2, 'v', 'velocity'}
            type = 'v'; typeFull = 'velocity';
        case {3, 'a', 'acceleration'}
            type = 'a'; typeFull = 'acceleration';
        otherwise
    end

    % Sort vectors
    [t, idx] = sort(t);
    y = y(idx);

    % If not given, estimate initial value for noise
    if ~exist('noiseStdev', 'var') || isempty(noiseStdev)
        if license('test', 'wavelet_toolbox')
            yDenoise = wdenoise(y);
        else
            yDenoise = smoothdata(y);
        end
        noiseStdev = std(y - yDenoise);
    end
    
    % Estimate spectrum: get time-step of training data
    dtVec = t(2:end) - t(1:end-1);
    if std(dtVec) == 0 % dt is constant
        newDt = dtVec(1);
        yInt = y;
    else
        newDt = min(dtVec);
        tInt = min(t):newDt:max(t);
        yInt = interp1(t, y, tInt);
    end    
    [~, fVec, ~, sVec] = Functions.PSD(yInt, newDt);

    % If mode isn't given, then we assumed it's a SDOF
    if ~exist('mode', 'var') || isempty(mode)   
        mode = 1;
    end

    % Add training data
    if isempty(self.trainingData)
        self.trainingData = struct; idx = 0;
    else
        idx = length(self.trainingData);
    end
    self.trainingData(idx+1).type = type;
    self.trainingData(idx+1).t = t;
    self.trainingData(idx+1).y = y;
    self.trainingData(idx+1).stdy = std(y);
    self.trainingData(idx+1).noise = (noiseStdev + eps);   
    self.trainingData(idx+1).mode = mode;  
    self.trainingData(idx+1).fVec = fVec; 
    self.trainingData(idx+1).sVec = sVec; 
    
    % Verbose
    if self.verbose > 1
        fprintf('Added %12s training data for mode %1.0f (%1.0f steps with avg dt = %1.2e, rms = %1.2e and noise sigma = %1.2e)\n', ...
            typeFull, mode, length(t), newDt, std(y), noiseStdev + eps);
    end
end
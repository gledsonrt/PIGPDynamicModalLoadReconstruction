function [MuF,SigF,Results] = predict(self, t)
    % Prediction of forces using the trained model
    
    % Check time vector
    assert(any(size(t) == 1), 'Time should be a vector');
    t = t(:);
    
    % Check available training data
    Nmodes = length(self.modeList);
    MuU = zeros(Nmodes, length(t));
    MuV = zeros(Nmodes, length(t));
    MuA = zeros(Nmodes, length(t));
    MuF = zeros(Nmodes, length(t));
    SigU = zeros(Nmodes, length(t));
    SigV = zeros(Nmodes, length(t));
    SigA = zeros(Nmodes, length(t));
    SigF = zeros(Nmodes, length(t));
    
    % Split time in chunks of 500, for better prediction speed
    tIDX = 1:500:length(t);
    if tIDX(end) ~= t(end); tIDX = [tIDX, length(t)]; end    

    % Loop over modal responses
    if self.verbose > 0; fprintf('Starting prediction of forces\n'); end
    modalCount = 1;
    for k = 1:length(tIDX)-1
        if k == length(tIDX)-1
            thisTIDX = tIDX(k):tIDX(k+1);
        else
            thisTIDX = tIDX(k):tIDX(k+1)-1;
        end
        thisT = t(thisTIDX);
        for i = 1:Nmodes
            
            % Verbose
            if self.verbose > 1 && i == modalCount
                fprintf('Now predicting mode %1.0f\n', self.modeList(i));
                modalCount = modalCount + 1;
            end

            % Get modal locations at structures
            trainingDataIDX = find(self.modeList(i) == [self.trainingData(:).mode]);

            % Basis functions for current time vector
            tDataTypes = [self.trainingData(trainingDataIDX).type];         
            if contains(tDataTypes, 'u')
                idx = find(tDataTypes == 'u');
                phiPU = self.generateBasisFunction(thisT, self.trainingData(trainingDataIDX(idx)).fVec, self.trainingData(trainingDataIDX(idx)).sVec, self.trainingData(trainingDataIDX(idx)).type);     
            else
                if contains(tDataTypes, 'v')
                    idx = find(tDataTypes == 'v');
                    phiPU = self.generateBasisFunction(thisT, self.trainingData(trainingDataIDX(idx)).fVec, {-1, self.trainingData(trainingDataIDX(idx)).sVec}, 'v');     
                else
                    idx = find(tDataTypes == 'a');
                    phiPU = self.generateBasisFunction(thisT, self.trainingData(trainingDataIDX(idx)).fVec, {-2, self.trainingData(trainingDataIDX(idx)).sVec}, 'a');     
                end
            end            
            if contains(tDataTypes, 'v')
                idx = find(tDataTypes == 'v');
                phiPV = self.generateBasisFunction(thisT, self.trainingData(trainingDataIDX(idx)).fVec, self.trainingData(trainingDataIDX(idx)).sVec, self.trainingData(trainingDataIDX(idx)).type);     
            else
                if contains(tDataTypes, 'u')
                    idx = find(tDataTypes == 'u');
                    phiPV = self.generateBasisFunction(thisT, self.trainingData(trainingDataIDX(idx)).fVec, {+1, self.trainingData(trainingDataIDX(idx)).sVec}, 'u');     
                else
                    idx = find(tDataTypes == 'a');
                    phiPV = self.generateBasisFunction(thisT, self.trainingData(trainingDataIDX(idx)).fVec, {-1, self.trainingData(trainingDataIDX(idx)).sVec}, 'a');     
                end
            end            
            if contains(tDataTypes, 'a')
                idx = find(tDataTypes == 'a');
                phiPA = self.generateBasisFunction(thisT, self.trainingData(trainingDataIDX(idx)).fVec, self.trainingData(trainingDataIDX(idx)).sVec, self.trainingData(trainingDataIDX(idx)).type);     
            else
                if contains(tDataTypes, 'v')
                    idx = find(tDataTypes == 'v');
                    phiPA = self.generateBasisFunction(thisT, self.trainingData(trainingDataIDX(idx)).fVec, {+1, self.trainingData(trainingDataIDX(idx)).sVec}, 'v');     
                else
                    idx = find(tDataTypes == 'u');
                    phiPA = self.generateBasisFunction(thisT, self.trainingData(trainingDataIDX(idx)).fVec, {+2, self.trainingData(trainingDataIDX(idx)).sVec}, 'u');     
                end
            end
            
            % Force kernel
            phiPF = self.M(i,i)*phiPA + self.C(i,i)*phiPV + self.K(i,i)*phiPU;

            % Build cross-covariance kernels
            crossKA = phiPA*self.kernels{i}{1,2}.*exp(self.hypsOpt{i}(1))^2;
            crossKV = phiPV*self.kernels{i}{1,2}.*exp(self.hypsOpt{i}(1))^2;
            crossKU = phiPU*self.kernels{i}{1,2}.*exp(self.hypsOpt{i}(1))^2;
            crossKF = phiPF*self.kernels{i}{1,2}.*exp(self.hypsOpt{i}(1))^2;
            
            % Displacements
            MuU(i,thisTIDX) = crossKU*self.alpha{i};
            covKU = (phiPU*phiPU').*exp(self.hypsOpt{i}(1))^2;
            SigU(i,thisTIDX) = sqrt(abs(diag(covKU - crossKU*(self.kinv{i}*crossKU'))))';
            
            % Velocities
            MuV(i,thisTIDX) = crossKV*self.alpha{i};
            covKV = (phiPV*phiPV').*exp(self.hypsOpt{i}(1))^2;
            SigV(i,thisTIDX) = sqrt(abs(diag(covKV - crossKV*(self.kinv{i}*crossKV'))))';
            
            % Accelerations
            MuA(i,thisTIDX) = crossKA*self.alpha{i};
            covKA = (phiPA*phiPA').*exp(self.hypsOpt{i}(1))^2;
            SigA(i,thisTIDX) = sqrt(abs(diag(covKA - crossKA*(self.kinv{i}*crossKA'))))';
            
            % Forces
            MuF(i,thisTIDX) = crossKF*self.alpha{i};
            covKF = (phiPF*phiPF').*exp(self.hypsOpt{i}(1))^2;
            SigF(i,thisTIDX) = sqrt(abs(diag(covKF - crossKF*(self.kinv{i}*crossKF'))))';

        end
    end

    % Extended results
    Results = struct;
    if nargout == 3
        Results.MuU = MuU;
        Results.SigU = SigU;
        Results.MuV = MuV;
        Results.SigV = SigV;
        Results.MuA = MuA;
        Results.SigA = SigA;
        Results.MuF = MuF;
        Results.SigF = SigF;
    end

end
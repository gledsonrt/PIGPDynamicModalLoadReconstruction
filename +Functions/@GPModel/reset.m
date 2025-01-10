function reset(self)
    % Brings GP model back to original state, previous to training
    self.kernels = [];
    self.trainingData = [];
    self.hyps0 = [];
    self.hypsOpt = [];
    self.alpha = [];
    self.kinv = [];
end
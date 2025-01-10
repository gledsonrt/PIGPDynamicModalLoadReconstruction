function [NLML, dNLML] = likelihood(self, params, modeNum)
    % Calculated the negative log marginal likelihood of the GP model based
    % on the training parameters and returns its value and the derivatives
    % w.r.t each of the scalars in [params]
    warning off;

    % Start outputs
    NLML = Inf;
    dNLML = zeros(size(params));

    % Check modes
    trainingModeList = find(self.modeList(modeNum) == [self.trainingData.mode]);
    
    % Build output vector
    y = [];
    for i = 1:length(trainingModeList)
        y = cat(1, y, self.trainingData(trainingModeList(i)).y);
    end
    
    % Number of training points
    n = length(y);
    
    % Build kernels
    Knoise = [];
    for i = 1:length(trainingModeList)
        Knoise = cat(1, Knoise, exp(params(i+1))^2*ones(size(self.trainingData(trainingModeList(i)).t)));
    end
    Knoise = Knoise + self.jitter;
    
    % Check for Woodbury matrix inversion
    if ~self.woodburyFlag
        % Standard GP on time-domain
        Kbase = self.kernels{modeNum}{1,3};
        Kdata = Kbase.*exp(params(1))^2;

        K = Kdata + diag(Knoise);

        % Cholesky decomposition
        [L,p] = chol(K, 'lower');
        if p > 0; return; end

        % Negative log likelihood
        Kinv = L'\(L\eye(n));
        alpha = Kinv*y;
        Kdet = 2*sum(log(diag(L)));
    else
        % Get basis function
        U = self.kernels{modeNum}{1,1};
        V = self.kernels{modeNum}{1,2};
        
        % Base kernel for derivatives
        Kbase = self.kernels{modeNum}{1,3}; %U*V;
        
        % Woodbury matrix decompositions
        Ainv = diag(1./Knoise);
        C = eye(size(U,2)).*exp(params(1))^2;

        % Inner part: needed for efficient determinant calculation
        VAinv = (V*Ainv);
        Kin = diag(1./diag(C)) + VAinv*U;
        
        % Cholesky decomposition
        [L,p] = chol(Kin, 'lower');
        if p > 0; return; end

        % Negative log likelihood
        Kinv = Ainv - (Ainv*U)*(L'\(L\VAinv));
        alpha = Kinv*y;
        Kdet = sum(log(Knoise)) + sum(log(diag(C))) + 2*sum(log(diag(L)));
    end
    
    % Negative log marginal likelihood
    dataFit = -0.5*y'*alpha;
    modelComplex = -0.5*Kdet;
    constant = -0.5*log(2*pi)*n;
    NLML = -(dataFit + modelComplex + constant);  
    
    % Store alpha
    self.alpha{modeNum} = alpha;
    self.kinv{modeNum} = Kinv;
    
    % Derivatives from log-likelihood
    if nargout > 1 && self.analytDerivs && ~self.woodburyFlag
        % Weights
        wSet = alpha*alpha' - Kinv;

        % Derivative w.r.t. sigma
        dSig = 2*exp(params(1)).*Kbase;
        dNLML(1) = -0.5*trace(wSet*dSig);

        % Derivative w.r.t. noise
        for i = 2:length(params)
            dNoise = zeros(n,1); m = 1;
            for j = 1:length(trainingModeList)
                if j == (i-1)
                    dNoise(m:m+length(self.trainingData(trainingModeList(j)).t)-1) = 2*exp(params(i));
                end
                m = m + length(self.trainingData(trainingModeList(j)).t);
            end
            dNLML(i) = -0.5*sum(diag(wSet).*dNoise);
        end
    end
end
classdef SDOF < handle
    properties
        m, c, k
        zeta, f, omega, T
        dt, Nsteps, SNR
        mode
    end
    methods
        % Constructor
        function self = SDOF(varargin)
            % Parse inputs
            p = inputParser;
            addOptional(p, 'm', 1);
            addOptional(p, 'c', NaN);
            addOptional(p, 'k', NaN);
            addOptional(p, 'zeta', NaN);
            addOptional(p, 'f', NaN);
            addOptional(p, 'dt', NaN);
            addOptional(p, 'SNR', NaN);
            parse(p, varargin{:});
            fNames = fieldnames(p.Results);
            for i = 1:length(fNames)
                eval(sprintf("self.%s = p.Results.%s;", fNames{i}, fNames{i}));
            end
            
            % Evaluate
            if isnan(self.k) && isnan(self.f)
                error('You need to specify either the stiffness or the natural frequency.'); 
            end
            if isnan(self.c) && isnan(self.zeta)
                error('You need to specify either the damping or the damping ratio to critical.'); 
            end
            
            % Calculate model properties
            if isnan(self.k)
                self.omega = 2*pi*self.f;
                self.k = self.m*self.omega^2;
            elseif isnan(self.f)
                self.omega = sqrt(self.k/self.m);
                self.f = self.omega/(2*pi);
            end
            self.T = 1/self.f;
            if isnan(self.c)
                self.c = 2*self.omega*self.m*self.zeta;
            else
                self.zeta = self.c/(2*self.omega*self.m);
            end
            [self.mode, ~] = eig(self.k, self.m);
            %disp(temp);
            
            % Integrator
            if isnan(self.dt)
                self.dt = self.T/20;
            end
        end
        
        function res = integrate(self, force, u0)
            % Get number of steps
            self.Nsteps = length(force);
            
            % Start response matrix
            res = zeros(self.Nsteps, 3);
            
            % Initial displacement
            if exist('u0', 'var') && ~isempty(u0)
                res(1,1) = u0;
                res(1,3) = -u0*(2*pi*self.f)^2;
            end
            
            % Time integrator
            for i = 2:self.Nsteps
                [res(i,1), res(i,2), res(i,3)] = Functions.NewmarkBeta(force(i), res(i-1,1), res(i-1,2), res(i-1,3), ...
                                                                             self.dt, self.m, self.c, self.k);
            end
            
            % Add noise: simulate sensor readings
            if ~isnan(self.SNR)
                self.SNR = abs(self.SNR) + eps;
                res(:,1) = awgn(res(:,1), self.SNR);
                res(:,2) = awgn(res(:,2), self.SNR);
                res(:,3) = awgn(res(:,3), self.SNR);
            end
        end
        
        function tVec = getTimeVector(self)
            if isempty(self.Nsteps) || isnan(self.Nsteps)
                error('Model has not been evaluated yet.')
            end
            tVec = (0:self.dt:self.dt*(self.Nsteps-1))';
        end
    end
end
function [phi] = generateBasisFunction(~, t, f, spec, type)
    % Generates the kernels for training based on data added to the GP
    % model. Basis functions are generated using the specified frequency
    % vector and spectrum
    
    % Is the spectrum given?
    if ~iscell(spec)
        % If spec is not a cell, then it's the actual spectrum for the
        % basis functions: decide on type for sign
        switch type
            case 'u' 
                phi = [sin(t*2*pi*f'), cos(t*2*pi*f')]*diag([spec; spec]);
            case 'v'
                phi = [cos(t*2*pi*f'), -sin(t*2*pi*f')]*diag([spec; spec]);
            case 'a'
                phi = [-sin(t*2*pi*f'), -cos(t*2*pi*f')]*diag([spec; spec]);
        end
    else
        % Is spec is a cell array, then the first value is a derivative
        % factor, the second value is the base spectrum
        fact = spec{1};
        spec = spec{2};
        derivMat = diag((2*pi*[f;f]).^fact);
        derivMat(derivMat == Inf) = 0;
        switch fact
            case -2
                phase = 0;
            case -1
                if strcmp(type, 'v')
                    phase = 0;
                elseif strcmp(type, 'a')
                    phase = +pi/2;
                end
            case +1
                if strcmp(type, 'u')
                    phase = +pi/2;
                elseif strcmp(type, 'v')
                    phase = +pi;
                end
            case +2
                phase = +pi;
        end
        phi = [sin(t*2*pi*f' + phase), cos(t*2*pi*f' + phase)]*diag([spec; spec])*derivMat;
    end
end
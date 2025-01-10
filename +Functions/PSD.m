function [psdSig, fVec, rms, dft_sig, dft_sig2] = PSD(sigA, sigB, dt)
    % Two-sided power spectral density via fast Fourier transform
    % Gledson Tondo
    warning off;

    % Check if dt exists: usually if not then it's a single signal
    if ~exist('dt', 'var') || isempty(dt)
        dt = sigB;
    
        % Lines as steps
        [s1,s2] = size(sigA);
        if s2 > s1; sigA = sigA'; end
        [s1,~] = size(sigA);

        % Time properties
        fs = 1/dt;

        % Fourier transform of signals
        % sig = sig - mean(sig, 1);
        dft_sig = fft(sigA, [], 1);
        dft_sig = dft_sig(1:s1/2+1,:);
        dft_sig2 = dft_sig(1:s1/2+1,:);
        psdSig = (1/(fs*s1)) * abs(dft_sig).^2;
        psdSig(2:end-1) = 2*psdSig(2:end-1);
        dft_sig = abs(dft_sig)/length(sigA); 
        dft_sig = 2*dft_sig;

        % Frequency vector
        fVec = (0:fs/s1:fs/2)';

        % RMS
        rms = sqrt(sum(psdSig,1)/(dt*s1));
    else
        % Lines as steps
        [s1A,s2A] = size(sigA);
        if s2A > s1A; sigA = sigA'; end
        [s1A,~] = size(sigA);
        [s1B,s2B] = size(sigB);
        if s2B > s1B; sigB = sigB'; end
        [s1B,~] = size(sigB);

        % Pad signals
        if s1A > s1B
            N = s1A;
            sigB = [sigB; zeros(s1A-s1B,1)];
        elseif s1B > s1A
            N = s1B;
            sigA = [sigA; zeros(s1B-s1A,1)];
        end

        % Time properties
        fs = 1/dt;

        % Fourier transform of signals
        dft_sigA = fft(sigA, [], 1);
        dft_sigA = dft_sigA(1:N/2+1,:);
        dft_sigB = fft(sigB, [], 1);
        dft_sigB = dft_sigB(1:N/2+1,:);
        psdSig = (1/(fs*N)) * abs(dft_sigA) .* abs(dft_sigB);
        psdSig(2:end-1) = 2*psdSig(2:end-1);
        dft_sig = abs(dft_sigA)/N;  
        dft_sig = 2*dft_sig;
        dft_sig2 = abs(dft_sigB)/N; 
        dft_sig2 = 2*dft_sig2;

        % Frequency vector
        fVec = (0:fs/N:fs/2)';

        % RMS
        rms = sqrt(sum(psdSig,1)/(dt*N));
    end
    
    warning on;
end
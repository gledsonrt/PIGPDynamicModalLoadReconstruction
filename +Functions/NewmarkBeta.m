function [u1, udot1, u2dot1] = NewmarkBeta(p, u, udot, u2dot, dt, m, c, k)
    % Newmark Method for SDOF Systems
    % Gledson Tondo
    
    % Parameters
    gamma = 0.5;
    beta = 0.25;

    % Coefficients
    a1 = 1/(beta.*dt^2).*m+gamma./(beta.*dt).*c;
    a2 = 1/(beta.*dt).*m+(gamma./beta-1).*c;
    a3 = (1/(2.*beta)-1).*m+dt.*(gamma./(2.*beta)-1).*c;
    kbar = k+a1;

    % Next time step
    pbar = p+a1.*u+a2.*udot+a3.*u2dot;
    u1 = pbar./kbar;
    udot1 = gamma./(beta.*dt).*(u1-u)+(1-gamma./beta).*udot+dt.*(1-gamma./(2.*beta)).*u2dot;     
    u2dot1 = 1./(beta.*dt.^2)*(u1-u)-1./(beta.*dt).*udot-(1./(2.*beta)-1)*u2dot;
end
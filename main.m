clc; close all; %clear all;

%rng('default')

N = 256; M = N/2;                   % N : # of transmit antennas, M : # of receive antennas

SNR= 0:4:24;

test_iter = 100;
jmax = 100;                          % Maximum number of iterations of the algorithm

symbols = qammod(linspace(0, 4-1, 4), 4,'UnitAveragePower',true);
symbols = [symbols, 0+0*1i];
unit_pow = abs(qammod(1,4));

Q = log2(length(symbols)-1);          % # of bits / symbol
bits = de2bi(0:length(symbols)-2,Q,'left-msb');

% Variables for storing NSERs
Nser_zf = zeros(length(SNR),1);
Nser_oracle = zeros(length(SNR),1);
Nser_ADMM_l2 = zeros(length(SNR),1);
Nser_cADMM = zeros(length(SNR),1);

% Variables for storing classification accuracys
Acc_zf = zeros(length(SNR),1);
Acc_oracle = zeros(length(SNR),1);
Acc_ADMM_l2 = zeros(length(SNR),1);
Acc_cADMM = zeros(length(SNR),1);

% Variables for storing simulation time
time_zf =  zeros(length(SNR),1);
time_oracle = zeros(length(SNR),1);
time_ADMM_l2 = zeros(length(SNR),1);
time_cADMM = zeros(length(SNR),1);

% Variables for storing iteration error
iter_oracle = 0;    iter_ADMM_l2 = 0;       iter_cADMM = 0;

for snr_pt = 1 : length(SNR)
    NSER_error_zf= 0;   NSER_error_oracle = 0;  NSER_error_ADMM_l2 = 0; NSER_error_cADMM = 0;   
    Acc_error_zf= 0;    Acc_error_oracle = 0;	Acc_error_ADMM_l2 = 0;	Acc_error_cADMM = 0;	
    
    num_active = 0;
    for simulation_no = 1 : test_iter
        % System model
        % Generate bits to be transmitted
        s_c = qammod( randi([0 4-1],N,1) ,4,'UnitAveragePower',true);
        
        % Random probability
        p = 0.1 * ones(N,1);
        lambda = log((1-p)./(p/4));   % Probability of signal generation
        
        s_c = binornd(1,p) .* s_c;
        
        active=find(s_c);                         	% support set
        if isempty(find(s_c, 1))
            continue;
        end
        
        % Generate the noise vector
        sigma = sqrt(N*10 .^ (-(SNR(snr_pt)/10)));    	% SNR points, in dB.
        sigma2 = sigma^2;
        
        noise = sigma/unit_pow*(randn(M,1) + 1i*randn(M,1));
        
        % Generate the random channel
        H_c = (randn(M,N) + 1i*randn(M,N))/unit_pow;
        H_oracle = H_c(:,active);
        H_bar = [H_c ; sigma*diag(sqrt(lambda))];
        
        % Transmit over noisy channel
        y_c = H_c*s_c + noise;
        y0 = [y_c ; zeros(N,1)];
        
        % Detector
        % ZF_Oracle detector
        zf_oracle = zeros(N,1);
        tic;
        s_oracle_zf = H_oracle \ y_c;
        zf_oracle(active) = qammod( qamdemod( (s_oracle_zf) ,4,'UnitAveragePower',true),4,'UnitAveragePower',true);
        time_oracle(snr_pt) = time_oracle(snr_pt)+toc;
        
        % ZF detector
        zf_result = zeros(N,1);
        tic
        s_zf = H_c\y_c;
        for i = 1: N
            if( abs(real((s_zf(i)))) + abs(imag((s_zf(i)))) > 1/unit_pow )
                zf_result(i) = qammod( qamdemod( (s_zf(i)) ,4,'UnitAveragePower',true),4,'UnitAveragePower',true);
            else
                zf_result(i) = 0;
            end
        end
        time_zf(snr_pt) = time_zf(snr_pt)+toc;
        
        % L2-norm ADMM detector
        tic;
        [ADMM_L2, iter_error_ADMM_L2] = ADMM_l2(y0, H_bar, 4, M, jmax, 1.7, lambda, sigma, unit_pow);
        time_ADMM_l2(snr_pt) = time_ADMM_l2(snr_pt)+toc;
        
        % Consensus-ADMM detector
        tic;
        [cADMM, iter_error_cADMM] = c_ADMM(y_c, H_c, 4, M, jmax, lambda, sigma, unit_pow);
        time_cADMM(snr_pt) = time_cADMM(snr_pt)+toc;
        
        % Error counting
        NSER_error_zf       = NSER_error_zf      + sum( (zf_result(active)) ~= (s_c(active)));
        NSER_error_oracle   = NSER_error_oracle  + sum( (zf_oracle(active)) ~= (s_c(active)));
        NSER_error_ADMM_l2  = NSER_error_ADMM_l2 + sum( (ADMM_L2(active))   ~= (s_c(active)));
        NSER_error_cADMM    = NSER_error_cADMM   + sum( (cADMM(active))     ~= (s_c(active)));
        
        % Calculate classification accuracys
        Acc_error_zf      = Acc_error_zf      + sum(length(intersect(find(zf_result),active))...
            + N-length(active)-length(setdiff(find(zf_result),active)) );
        Acc_error_oracle  = Acc_error_oracle  + sum(length(intersect(find(zf_oracle),active))...
            + N-length(active)-length(setdiff(find(zf_oracle),active)) );
        Acc_error_ADMM_l2 = Acc_error_ADMM_l2 + sum(length(intersect(find(ADMM_L2),active))...
            + N-length(active)-length(setdiff(find(ADMM_L2),active)) );
        Acc_error_cADMM   = Acc_error_cADMM   + sum(length(intersect(find(cADMM),active))...
            + N-length(active)-length(setdiff(find(cADMM),active)) );
        
        % Trajectory of objective values with respect to iterations at last SNR
        if snr_pt == length(SNR)
            iter_oracle   = iter_oracle  + norm(y_c-H_oracle*zf_oracle(active))^2 + sigma^2*sum(lambda.*abs(zf_oracle));
            iter_ADMM_l2  = iter_ADMM_l2 + iter_error_ADMM_L2;
            iter_cADMM    = iter_cADMM   + iter_error_cADMM;
        end
        
        if mod(simulation_no,100) == 0
            fprintf('SNR :%d \t iteration : %d \t\t'+string(datetime)+'\n', SNR(snr_pt), simulation_no);
        end
        
        num_active = num_active + length(active);
    end
    
    % NSER Normalization
    Nser_zf(snr_pt)     = NSER_error_zf      / (num_active);
    Nser_oracle(snr_pt) = NSER_error_oracle  / (num_active);
    Nser_ADMM_l2(snr_pt)= NSER_error_ADMM_l2 / (num_active);
    Nser_cADMM(snr_pt)  = NSER_error_cADMM   / (num_active);
    
    % Accuracys Normalization
    Acc_zf(snr_pt)     = Acc_error_zf      / (N * test_iter) * 100;
    Acc_oracle(snr_pt) = Acc_error_oracle  / (N * test_iter) * 100;
    Acc_ADMM_l2(snr_pt)= Acc_error_ADMM_l2 / (N * test_iter) * 100;
    Acc_cADMM(snr_pt)  = Acc_error_cADMM   / (N * test_iter) * 100;
    
    % Time Normalization
    time_zf(snr_pt)     = time_zf(snr_pt)      / (test_iter);
    time_oracle(snr_pt) = time_oracle(snr_pt)  / (test_iter);
    time_ADMM_l2(snr_pt)= time_ADMM_l2(snr_pt) / (test_iter);
    time_cADMM(snr_pt)  = time_cADMM(snr_pt)   / (test_iter);
end


% Iteration error Normalization
iter_oracle = iter_oracle  / test_iter;
iter_ADMM_l2= iter_ADMM_l2 / test_iter;
iter_cADMM  = iter_cADMM   / test_iter;

% plot SNR vs SER graph
figure;
semilogy(SNR(1:length(SNR)),Nser_zf(1:length(SNR)),'-.^','Color',[0 102/255 102/255]); hold on;
semilogy(SNR(1:length(SNR)),Nser_ADMM_l2(1:length(SNR)),'--bx');
semilogy(SNR(1:length(SNR)),Nser_cADMM(1:length(SNR)),'-ro');
semilogy(SNR(1:length(SNR)),Nser_oracle(1:length(SNR)),'-.k*');
legend('ZF','ADMM(L2-Regularized)','Consensus-ADMM','Oracle-ZF','location','southwest');
xlabel('SNR (dB)'); ylabel('NSER'); grid;
axis([0 24 -inf inf]);

% plot SNR vs classification accuracys graph
figure
plot(SNR,Acc_oracle,'-.k');  hold on; 
plot(SNR,Acc_cADMM,'-ro');
plot(SNR,Acc_ADMM_l2,'--bx');
plot(SNR,Acc_zf,'-d','Color',[0 102/255 102/255]);
legend('Oracle-ZF','Consensus-ADMM','ADMM(L2-Regularized)','ZF','location','southeast');
xlabel('SNR (dB)'); ylabel('Classification Accuracy (%)'); grid;
axis([0 24 88 inf])

% plot SNR vs iteration error graph
figure
semilogy(1:jmax,iter_ADMM_l2(1:jmax),'--b');  hold on;
semilogy(1:jmax,iter_cADMM(1:jmax),'r');
semilogy(1:jmax,iter_oracle*ones(1,jmax),'-.k');
legend('ADMM(L2-Regularized)','Consensus-ADMM','Oracle-ZF');
xlabel('number of Iterations'); ylabel('\Sigma_{i=1}^2 f_i^*(x^{(k)})'); grid;
axis([0 50 100 inf])

% Simulation time
fprintf('Simulation time \n\nZF\t\t: %f \nADMM_l2\t: %f \ncADMM\t: %f \nOracle\t: %f \n',...
    time_zf(length(SNR)), time_ADMM_l2(length(SNR)), time_cADMM(length(SNR)), time_oracle(length(SNR)))



function [x_hat, iter_error] = c_ADMM( r, H, Mod, rho, jmax, lambda, sigma, pow)
% Consensus-ADMM
N = size(H,2);

u1 = zeros(N,1); u2 = zeros(N,1); u3 = zeros(N,1);
x3 = zeros(N,1);
x_bar = zeros(N,1);

ABSTOL = 1e-4;
eps = sqrt(N)*ABSTOL;
RELTOL = 1e-4;

result_x = zeros(N,jmax);
result_res_pri = zeros(1,jmax);
iter_error = ones(1,jmax);

H_inv = (H'*H + rho/2*eye(N));
Hr = H'*r;

for iter = 1:jmax
    
    x_bar_old = x_bar;
    
    x1 = H_inv\(Hr + rho/2*(x_bar-u1));
    
    x2 = prox(x_bar-u2, (lambda*sigma^2)/rho);
    
    for i = 1: N
        if( abs(real((x_bar(i)-u3(i)))) + abs(imag((x_bar(i)-u3(i)))) > 1/pow)
            x3(i) = qammod(qamdemod(x_bar(i)-u3(i),Mod,'UnitAveragePower',true),Mod,'UnitAveragePower',true);
        else
            x3(i) = 0;
        end
    end
    
    x_bar = (x1+x2+x3)/3;
    
    % lagrangian mutiplier update
    u1 = u1 + x1 - x_bar;
    u2 = u2 + x2 - x_bar;
    u3 = u3 + x3 - x_bar;
    
    % find minimize value
    result_x(:,iter) = x3;
    result_res_pri(iter) = norm(r-H*x3)^2 + sigma^2*sum(lambda.*abs(x3));
  
    % compute threshold
    pri = [x1 - x_bar;x2 - x_bar;x3 - x_bar];
    X   = [x1;x2;x3];
    U   = [u1;u2;u3];
    
    % compute residual
    res_pri = norm(pri,'fro');
    res_dual = norm((x_bar-x_bar_old)*rho);
    
    % compute threshold
    eps_pri = eps + RELTOL*max(norm(X,'fro'), norm(x_bar));
    eps_dual = eps + RELTOL*rho*norm(U,'fro');
    
    
    [~, index]=min(result_res_pri(1:iter));
    iter_error(iter)=result_res_pri(index);
    
%     check termination condition
        if res_pri<eps_pri && res_dual<eps_dual   
            iter_error(iter+1:jmax) = iter_error(iter).*ones(1,jmax-iter);
            break;
        end
end

x_hat = result_x(:,index);
end
function [s_hat, iter_error] = ADMM_l2( r, H, Mod, rho, jmax, alpha, lambda, sigma, pow)
%ADMM
if nargin<4
    rho = 10;
end
if nargin<5
    jmax = 100;
end
if nargin<6
    alpha = 1.4;
end
N = size(H,2);
u = zeros(N,1);
z = zeros(N,1);

H_inv = inv(H'*H + rho/2*eye(N));
Hr = H'*r;

ABSTOL = 1e-4;
eps = sqrt(N)*ABSTOL;
RELTOL = 1e-4;

result_z = zeros(N,jmax);
result_res_pri = zeros(1,jmax);
iter_error = ones(1,jmax);
for iter = 1:jmax
    x = H_inv*(Hr + rho/2*(z-u));
    z_old = z;
    x_hat = alpha*x + (1-alpha)*z_old;
    
    for i = 1: N
        if( abs(real((x_hat(i)+u(i)))) + abs(imag((x_hat(i)+u(i)))) > 1/pow)
            z(i) = qammod(qamdemod((x_hat(i)+u(i)),Mod,'UnitAveragePower',true),Mod,'UnitAveragePower',true);
        else
            z(i) = 0;
        end
    end
    u = u + x_hat - z;
      
    % find minimize value
    result_z(:,iter) = z;
    result_res_pri(iter) = norm(r-H*z)^2 + sigma^2*sum(lambda.*abs(z));
    
    % compute residual
    res_pri = norm(x-z);
    res_dual = norm((z-z_old)*rho);
    
    % compute threshold
    eps_pri = eps + RELTOL*max(norm(x), norm(-z));
    eps_dual = eps + RELTOL*rho*norm(u);
    
    [~, index]=min(result_res_pri(1:iter));
    iter_error(iter)=result_res_pri(index);
    
%     check termination condition
        if res_pri<eps_pri && res_dual<eps_dual    
            iter_error(iter+1:jmax) = iter_error(iter).*ones(1,jmax-iter);
            break;
        end
end

s_hat = result_z(:,index);
end
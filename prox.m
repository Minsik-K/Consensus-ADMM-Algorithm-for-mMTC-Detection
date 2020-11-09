function x = prox(v, gamma)
x= zeros(length(v),1);
for i = 1 : length(v)
    if abs(v(i)) <= gamma(i)
        x(i)=0;
    else
        x(i) = (1-gamma(i)/abs(v(i)))*v(i);
    end
end
end
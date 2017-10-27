function [ E ] = average(E)

E = mean(E, 1); 
T = size(E, 2);

for i=2:T-1
    if E(i-1) < E(i)
        E(i:end) = [];
        break;
    end
end


end


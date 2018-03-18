c = 3; % Number of medical students
k = floor(c/2)+1; % At least correct decisions
p = 0.8; % probability of individual making correct decision
Y = binopdf(k:c,c,p); % Accumulate binomial pdf per case 
sum(Y)


binopdf(1:3,3,0.8)

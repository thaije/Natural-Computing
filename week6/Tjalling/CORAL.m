function [Dsnew] = CORAL(Ds, Dt)
%Does Correlation Allignment

Cs = cov(Ds) + eye(size(Ds,2));
Ct = cov(Dt) + eye(size(Dt,2));

Ds = Ds * Cs^(-.5);
Dsnew = Ds * Ct^(0.5);

end


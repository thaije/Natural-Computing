function [fts] = prepro(fts)
% Does preprocessing on SURF data set
fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
fts = zscore(fts,1); 
end


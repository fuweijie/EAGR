function [Z] = FLAE(anchor, data, knn, beta)

% Fast Local Anchor Embedding 
% Written by Weijie Fu (fwj.edu@gmail.com)
if ~exist('knn', 'var') || isempty(knn),
    knn = 3;
end
if ~exist('beta', 'var') || isempty(beta),
    beta = 0;
end

n=size(data,1);
m=size(anchor,1);

% find k nearest neighbors   
D= sqdist(data',anchor');
IDX = zeros(n, knn);
for i = 1:n,
	d = D(i,:);
	[~, idx] = sort(d, 'ascend');
    IDX(i, :) = idx(1:knn);
end

II = eye(knn, knn);
Z = zeros(n, m);

% Local weight estimation
for i=1:n
   idx = IDX(i,:);
   z = anchor(idx,:) - repmat(data(i,:), knn, 1);           
   C = z*z'; 
   d=diag(C);
   d=exp(d/max(d));
   d=d./sum(d);
   C = C + beta*diag(d)+II*(1e-4);
   w = (C)\ones(knn,1);
   w = w/sum(w);
   w=abs(w);
   w=w/sum(w);
   Z(i,idx) = w';
end

Z=sparse(Z);

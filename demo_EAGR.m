% Demo of EAGR
% Detials in 'Scalable Semi-Supervised Learning by Efficient Anchor Graph Regularization'
clear
load('Data_Letter.mat')
m=500;
[IDX,anchor]=kmeans(data,m,'MaxIter',10,'emptyaction','singleton');

% Local weight estimation
[Z] = FLAE(anchor,data,3,1);

% Normalized graph Laplacian
W=Z'*Z;
Dt=diag(sum(W).^(-1/2));
S=Dt*W*Dt;
rL=eye(m,m)-S;

% Graph Regularization
accuracy=zeros(1,5);
for iter=1:5
    acc= EAGReg(Z,rL, label', label_index(iter,:));
    accuracy(iter) =acc;   
end
fprintf('\n The average classification accuracy of EAGR is %.2f%%.\n', mean(accuracy)*100);
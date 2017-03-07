function [acc] = EAGReg(Z, rL, label, label_index,gamma)

if ~exist('gamma', 'var') || isempty(gamma),
    gamma = 1;
end

[n,m]=size(Z);
ln = length(label_index);
C = max(label);

% Label matrix construction
Yl = zeros(ln,C);
for i = 1:C
    ind = find(label(label_index) == i);
    Yl(ind',i) = 1;
    clear ind;
end
Zl = Z(label_index',:);

% Regularization
LM = Zl'*Zl+gamma*rL;
RM = Zl'*Yl;  
A=inv(LM+1e-6*eye(m))*RM;  

% Label inference
F = Z*A; 
F = F*diag(sum(F).^-1);

[~,output] = max(F,[],2);
output=output';

%Error estimation
output(label_index) = label(label_index);
acc = 1-length(find(output ~= label))/(n-ln);



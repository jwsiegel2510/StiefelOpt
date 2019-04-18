function [ output ] = retract( X, V )
%RETRACT Calculates the Cayley retraction on the Stiefel manifold.

s = size(X);
U = [0.5*V, X];
W = [X, -0.5*V];


output = X + 2*U*mldivide(eye(2*s(2)) - W'*U, W'*X);
% Renormalize the columns (cheap and improves accuracy with which
% orthogonality contraint is preserved, though not really necessary)
output = output*spdiags([sum(output.^2).^(-1/2)'],[0],s(2),s(2));

end


function [ output ] = apply_momentum( X, Y, alpha )
%APPLY_MOMENTUM Applies momentum based on the Cayley retraction on the
% Stiefel manifold modulo negation of each column (use when considering
% problems with this particular type of symmetry).

s = size(X);
V_t = 2 * mldivide(eye(s(2)) + Y'*X, Y')';
V_t = V_t - 0.5 * X * (V_t' * X + X' * V_t);

output = retract(X, (1.0 + alpha) * V_t);

end


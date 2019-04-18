function [ A ] = InhomoLaplace( w, pot )
% Returns a sparse linear operator representing a periodic inhomogeneous 
% Laplacian with the potential pot.
   
s = size(w);
A = sparse(s(1)*s(2), s(1)*s(2));

% Construct appropriate linear operator.
for i = 0:s(1)-1
    for j = 0:s(2)-1
        A(j + s(2)*i + 1, j + s(2)*i + 1) = A(j + s(2)*i + 1, j + s(2)*i + 1) ...
            + (w(i + 1, j + 1) + w(mod(i + 1, s(1)) + 1, j + 1)) / 2.0;
        A(j + s(2)*i + 1, j + s(2)*mod(i + 1, s(1)) + 1) = A(j + s(2)*i + 1, j + s(2)*mod(i + 1, s(1)) + 1) ...
            - (w(i + 1, j + 1) + w(mod(i + 1, s(1)) + 1, j + 1)) / 2.0;
        A(j + s(2)*i + 1, j + s(2)*i + 1) = A(j + s(2)*i + 1, j + s(2)*i + 1) ...
            + (w(i + 1, j + 1) + w(mod(i - 1, s(1)) + 1, j + 1)) / 2.0;
        A(j + s(2)*i + 1, j + s(2)*mod(i - 1, s(1)) + 1) = A(j + s(2)*i + 1, j + s(2)*mod(i - 1, s(1)) + 1) ...
            - (w(i + 1, j + 1) + w(mod(i - 1, s(1)) + 1, j + 1)) / 2.0;
        A(j + s(2)*i + 1, j + s(2)*i + 1) = A(j + s(2)*i + 1, j + s(2)*i + 1) ...
            + (w(i + 1, j + 1) + w(i + 1, mod(j + 1, s(2)) + 1)) / 2.0;
        A(j + s(2)*i + 1, mod(j + 1, s(2)) + s(2)*i + 1) = A(j + s(2)*i + 1, mod(j + 1, s(2)) + s(2)*i + 1) ...
            - (w(i + 1, j + 1) + w(i + 1, mod(j + 1, s(2)) + 1)) / 2.0;
        A(j + s(2)*i + 1, j + s(2)*i + 1) = A(j + s(2)*i + 1, j + s(2)*i + 1) ...
            + (w(i + 1, j + 1) + w(i + 1, mod(j - 1, s(2)) + 1)) / 2.0;
        A(j + s(2)*i + 1, mod(j - 1, s(2)) + s(2)*i + 1) = A(j + s(2)*i + 1, mod(j - 1, s(2)) + s(2)*i + 1) ...
            - (w(i + 1, j + 1) + w(i + 1, mod(j - 1, s(2)) + 1)) / 2.0;
        A(j + s(2)*i + 1, j + s(2)*i + 1) = A(j + s(2)*i + 1, j + s(2)*i + 1) + pot(i+1,j+1);
    end
end


end


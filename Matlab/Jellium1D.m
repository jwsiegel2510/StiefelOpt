function it = Jellium1D
% Simulates an infinite slice of Jellium using DFT and optimization on the
% Stiefel manifold.

grid_disc = 5000; % Number of spatial grid points per unit.
k = 20; % Half of the number of electrons.
scale = 10; % The length of the simulation in atomic units.

n = 2*grid_disc + 1;

% Generate positive background charge
bc = zeros(n, 1);
width = floor(grid_disc * .333); % width is one third of total simulation size
bc(grid_disc + 1 - width: grid_disc + 1 + width) = ones(2*width + 1, 1);
bc = 2 * bc * k / (2*width + 1);

% Generate initial point at random on the Stiefel manifold.
R = random('Normal', zeros(n,k), ones(n,k));
[X,~] = qr(R,0);

% Create sparse second derivative matrix.
A = spdiags([-1*ones(n,1), 2*ones(n,1), -1*ones(n,1)], [-1, 0, 1], n, n) * grid_disc^2 / (scale*sqrt(8*pi));

[X,it] = stiefel_opt_function_restart(X, @obj, @grad, 0, 1e-3, A, bc, grid_disc);

J = 2 * X.^2*ones(k,1);
plot([-scale:scale/grid_disc:scale],J); hold on;
plot([-scale:scale/grid_disc:scale],bc);
end

function F = obj(X, A, bc, grid_disc)
    s = size(X);
    n = s(1);
    k = s(2);
    density = 2 * X.^2*ones(k,1);
    pot_left = -1.0*density + 2*bc;
    pot_right = -1.0*density + 2*bc;
    pot_left = cumsum(cumsum(pot_left)); % Calculate the potential due to charges on the left.
    pot_left = circshift(pot_left, 1);
    pot_left(1) = 0;
    pot_right = cumsum(cumsum(pot_right, 'reverse'), 'reverse'); % due to charges on the right.
    pot_right = circshift(pot_right, -1);
    pot_right(end) = 0;
    F = .5*trace(X'*A*X) + .5*dot(density, (pot_right + pot_left) / grid_disc);
end

function dF = grad(X, A, bc, grid_disc)
    s = size(X);
    n = s(1);
    k = s(2);
    density = 4 * X.^2 * ones(k,1);
    pot_left = -1.0*density + 2*bc;
    pot_right = -1.0*density + 2*bc;
    pot_left = cumsum(cumsum(pot_left)); % Calculate the potential due to charges on the left.
    pot_left = circshift(pot_left, 1);
    pot_left(1) = 0;
    pot_right = cumsum(cumsum(pot_right, 'reverse'), 'reverse'); % due to charges on the right.
    pot_right = circshift(pot_right, -1);
    pot_right(end) = 0;
    dF = A*X + 2*((pot_right + pot_left)*ones(1,k)) .* X / grid_disc;
end


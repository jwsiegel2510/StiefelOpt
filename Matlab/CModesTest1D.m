function obj_val = CModesTest1D

n = 1000;
k = 20;
mu = 0.2;

A = spdiags([-1.0*ones(n,1) 2.0*ones(n,1) -1.0*ones(n,1)], [-1,0,1], n , n) * n^2;

X = zeros(n,k);

for i = 1:k
    X((i-1)*(n/k) + 1: i*(n/k), i) = sqrt(k/n);
end

X = averaged_subgradient(X, @grad, 2.5 / n^2, 100000, A, mu*n);

obj_val = obj(X, A, mu*n);

for i=1:k
    plot([0:1/1000:1-1/1000], X(:,i)); hold on;
end

conv_data = [];
j = 50;
for i=1:20
    X = zeros(n,k);

    for l = 1:k
        X((l-1)*(n/k) + 1: l*(n/k), l) = sqrt(k/n);
    end
    
    X = averaged_subgradient(X, @grad, 2.5 / n^2, j, A, mu*n);
    
    conv_data = [conv_data [j (obj(X, A, mu*n) - obj_val)]'];
    j = floor(j*1.4);
end
figure; plot(log(conv_data(1,:)), log(conv_data(2,:)));
end

function dF = grad(X, A, mu)
    dF = A*X;
    dF = dF + mu*sign(X);
end

function F = obj(X, A, mu)
    F = .5*trace(X'*A*X) + mu*sum(sum(abs(X)));
end
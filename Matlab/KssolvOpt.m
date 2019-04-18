function output = KssolvOpt
sih4_setup;
Init = genX0(mol);

Ewald     = getewald(mol);
Ealphat   = getealphat(mol);

X = get(Init,'psi');
X = cat(2, X{:});

[X,it] = stiefel_opt_function_restart(X, @obj, @grad, 0, 1e-3, mol);

Etot = Ewald + Ealphat + obj(X, Init, mol);
output = [X Etot it];
end

function F = obj(X, mol)
    Init = genX0(mol);
    Xfun = Wavefun(X, get(Init,'n1'), get(Init, 'n2'), get(Init,'n3'), get(Init, 'idxnz'));
    H = Ham(mol);
    nspin = get(mol,'nspin');
   %
   % Kinetic energy and some additional energy terms 
   %
   Ekin = (2/nspin)*real(trace(Xfun'*applyKIEP(H,Xfun)));       
   %
   % Compute Hartree and exchange correlation energy and potential
   % using the new charge density; update the total potential
   %
   rho = getcharge(mol,Xfun);
   [vhart,~,uxc2,rho]=getvhxc(mol,rho);
   %
   % Calculate the potential energy based on the new potential
   %
   Ecoul = getEcoul(mol,abs(rho),vhart);
   Exc   = getExc(mol,abs(rho),uxc2);
   F = Ekin + Ecoul + Exc;
end

function df = grad(X, mol)
    Init = genX0(mol);
    Xfun = Wavefun(X, get(Init,'n1'), get(Init, 'n2'), get(Init,'n3'), get(Init, 'idxnz'));
    rhoout = getcharge(mol,Xfun);
    H = Ham(mol, rhoout);
    DfFun = H*Xfun;
    df = get(Init,'psi');
    df = cat(2, df{:});
end
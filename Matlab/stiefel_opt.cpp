/* Contains a Mex C++ implementation of accelerated gradient descent on the
Stiefel manifold for optimizing a Brockett cost. The function is passed 5 
arguments. The first is matrix whose eigenvectors are sought, the second is
a vector giving the Brockett weights, the third is the initial iterate, the
4th is gradient norm tolerance, and the 5th is a integer flag which
specifies the method to be used. 0 specifies gradient descent, 1 specifies
function restart accelerated gradient descent, and 2 specifies gradient
restart accelerated gradient descent. It is assumed that the initial 
iterate is in fact on the Stiefel manifold. This implementation 
makes use of the ROPTLIB library. */

#include "mex.h"

/*Libraries from ROPTLIB*/
#include "Manifolds/Stiefel/Stiefel.h"
#include "Manifolds/Stiefel/StieVariable.h"
#include "Manifolds/Stiefel/StieVector.h"
#include "Manifolds/Manifold.h"
#include "Problems/StieSparseBrockett/StieSparseBrockett.h"
#include "Problems/mexProblem.h"
#include "Others/def.h"

/*Output to console*/
#include <iostream>
#include <cmath>

/*Generate random number*/
#include "Others/randgen.h"

/*Computational time*/
#include <ctime>

using namespace ROPTLIB;

// Projects StieGrad onto the dual tangent space and returns its squared norm.
double ProjectGradAndReturnNormSq(const StieVariable* StieY, StieVector* StieGrad, double* symUtV, integer n, integer p) {
    const double *Y = StieY->ObtainReadData();
    double *G = StieGrad->ObtainWriteEntireData();
    
    char *transn = const_cast<char *> ("n"), *transt = const_cast<char *> ("t");
	double one = 1, zero = 0;
    double negone = -1;
    
    dgemm_(transt, transn, &p, &p, &n, &one, const_cast<double *> (G), &n, const_cast<double *> (Y), &n, &zero, symUtV, &p);

    double Gnormsq = 0;
    for (integer i = 0; i < p; i++)
    {
        for (integer j = i + 1; j < p; j++)
        {
            Gnormsq += 0.5 * (symUtV[i + j * p] - symUtV[j + i * p]) * (symUtV[i + j * p] - symUtV[j + i * p]);
            symUtV[i + j * p] = 0.5 * symUtV[i + j * p] + 0.5 * symUtV[j + i * p];
            symUtV[j + i * p] = symUtV[i + j * p];
        }
    }
    
    dgemm_(transn, transn, &n, &p, &p, &negone, const_cast<double *> (Y), &n, symUtV, &p, &one, G, &n);
    
    // Calculate squared norm.
    integer len = n*p; integer inc = 1;
    Gnormsq += ddot_(&len, const_cast<double *> (G), &inc, const_cast<double *> (G), &inc);
    
    return Gnormsq;
}

// Copies the first argument to the second.
void StieCopy(const StieVariable* StieX, StieVariable* StieY, integer n, integer p) {
    const double *StieXptr = StieX->ObtainReadData();
    double *StieYptr = StieY->ObtainWriteEntireData();
    
    integer length = n*p;
    integer inc = 1;
    dcopy_(&length, const_cast<double *> (StieXptr), &inc, StieYptr, &inc);
}

// Performs a Cayley Retraction from Y in the direction step_size * V and stores the
// output in X.
void retract(const StieVariable* Y, const StieVector* V, StieVariable* X, double step_size, double* YV, double* VY, double* YVVY, double* VYY, integer* pivots, integer n, integer p) {
    StieCopy(Y, X, n, p);
    const double *StieYptr = Y->ObtainReadData();
    const double *StieVptr = V->ObtainReadData();
    
    // Set YV = [Y, 0.5 * step_size * V].
    integer length = n*p; integer inc = 1;
    double factor = 0.5 * step_size;
    dcopy_(&length, const_cast<double *> (StieYptr), &inc, YV, &inc);
    dcopy_(&length, const_cast<double *> (StieVptr), &inc, YV + length, &inc);
    dscal_(&length, &factor, YV + length, &inc);
    
    // Set VY = [0.5 * step_size * V,-Y].
    double minus_one = -1.0; double one = 1.0;
    dcopy_(&length, const_cast<double *> (StieVptr), &inc, VY, &inc);
    dcopy_(&length, const_cast<double *> (StieYptr), &inc, VY + length, &inc);
    dscal_(&length, &minus_one, VY + length, &inc);
    dscal_(&length, &factor, VY, &inc);
    
    // Set YVVY = VY' * YV.
    double zero = 0;
    char *transn = const_cast<char *> ("n"), *transt = const_cast<char *> ("t");
    integer P = 2 * p; integer N = n;
    dgemm_(transt, transn, &P, &P, &N, &one, const_cast<double *> (VY), &N, const_cast<double *> (YV), &N, &zero, YVVY, &P);
    
    // Add the identity to YVVY.
    for (integer i = 0; i < 2 * p; ++i) YVVY[i + i * 2 * p] += 1.0;
    
    // Set VYY = VY' * Y.
    dgemm_(transt, transn, &P, &p, &N, &one, const_cast<double *> (VY), &N, const_cast<double *> (StieYptr), &N, &zero, VYY, &P);
    
    // Set VYY = (YVVY)^{-1} * VYY.
    integer info;
    dgesv_(&P, &p, YVVY, &P, pivots, VYY, &P, &info);
    
    // Set X = X - 2.0 * YV * VYY.
    double *StieXptr = X->ObtainWriteEntireData();
    double minus_two = -2.0;
    dgemm_(transn, transn, &N, &p, &P, &minus_two, const_cast<double *> (YV), &N, VYY, &P, &one, StieXptr, &N);
    
    if (info != 0) printf("Retraction Failed. \n");
}

// Calculates momentum vector from X_temp to X and stores the result in V.
void find_momentum_vector(const StieVariable* X_temp, const StieVariable* X, StieVector* V, double* symUtV, double* invMat, integer* pivots, integer n, integer p) {
    const double *StieXptr = X->ObtainReadData();
    const double *StieXtempptr = X_temp->ObtainReadData();
    
    char *transn = const_cast<char *> ("n"), *transt = const_cast<char *> ("t");
	double one = 1, zero = 0;
    double negone = -1; double two = 2.0;
    
    // Set SymUtV = X_temp^t * X.
    dgemm_(transt, transn, &p, &p, &n, &one, const_cast<double *> (StieXtempptr), &n, const_cast<double *> (StieXptr), &n, &zero, symUtV, &p);
    
    // Set invMat = id.
    integer length = p*p; integer inc = 1;
    dscal_(&length, &zero, invMat, &inc);
    for (integer i = 0; i < p; ++i) invMat[i + i*p] += 1.0;
    
    // Add the identity to SymUtV.
    for (integer i = 0; i < p; ++i) symUtV[i + i*p] += 1.0;
    
    // Set invMat = SymUtV^-1.
    integer info;
    dgesv_(&p, &p, symUtV, &p, pivots, invMat, &p, &info);
        
    if (info != 0) printf("Momentum failed.\n");
    
    // Set V = 2 * X * invMat.
    double* StieVptr = V->ObtainWriteEntireData();
    dgemm_(transn, transn, &n, &p, &p, &two, const_cast<double *> (StieXptr), &n, invMat, &p, &zero, StieVptr, &n);
    
    // Project V onto the dual tangent space at Xtemp.
    dgemm_(transt, transn, &p, &p, &n, &one, const_cast<double *> (StieVptr), &n, const_cast<double *> (StieXtempptr), &n, &zero, symUtV, &p);
    for (integer i = 0; i < p; i++)
    {
        for (integer j = i + 1; j < p; j++)
        {
            symUtV[i + j * p] = 0.5 * symUtV[i + j * p] + 0.5 * symUtV[j + i * p];
            symUtV[j + i * p] = symUtV[i + j * p];
        }
    }
    
    dgemm_(transn, transn, &n, &p, &p, &negone, const_cast<double *> (StieXtempptr), &n, symUtV, &p, &one, StieVptr, &n);
}

// Calculates the inner product between two dual tangent vectors.
double calculate_stie_inner_product(const StieVector* V, const StieVector* W, const StieVariable* StieY, double* VTU, double* VTUtemp, double* VTUtemp2, integer n, integer p) {
    const double *StieVptr = V->ObtainReadData();
    const double *StieWptr = W->ObtainReadData();
    const double *StieYptr = StieY->ObtainReadData();
    
    char *transn = const_cast<char *> ("n"), *transt = const_cast<char *> ("t");
	double one = 1, zero = 0;
    
    // Set VTU = V^T * StieY
    dgemm_(transt, transn, &p, &p, &n, &one, const_cast<double *> (StieVptr), &n, const_cast<double *> (StieYptr), &n, &zero, VTU, &p);
    
    // Set VTUtemp = StieY^T * W
    dgemm_(transt, transn, &p, &p, &n, &one, const_cast<double *> (StieYptr), &n, const_cast<double *> (StieWptr), &n, &zero, VTUtemp, &p);
    
    // Set VTUtemp2 = VTU * VTUtemp
    dgemm_(transn, transn, &p, &p, &p, &one, VTU, &p, VTUtemp, &p, &zero, VTUtemp2, &p);
    
    // Set VTU = V^T * W
    dgemm_(transt, transn, &p, &p, &n, &one, const_cast<double *> (StieVptr), &n, const_cast<double *> (StieWptr), &n, &zero, VTU, &p);
    
    // Collect results
    double output = 0;
    for (integer i = 0; i < p; ++i) {
        output += VTUtemp2[i + i * p];
        output += VTU[i + i * p];
    }
    return output;
}

int GradDescent(StieVariable* StieX, Problem* Prob, double tol, integer n, integer p) {
    // Parameter controlling the decrease factor for the armijo search.
    const double alpha = 1.7;

    // Allocate temp memory for future calculations.
    
    double *symUtV = new double[p * p];
    double *VY = new double[n * p * 2];
    double *YV = new double[n * p * 2];
    double *YVVY = new double[4 * p * p];
    double *VYY = new double[2 * p * p];
    integer* pivots = new integer[2 * p];
    StieVariable StieXtemp(n,p);
    StieVector StieGrad(n,p);
    
    double step_size = 0.1;
    double FX = Prob->f(StieX);
    double FXtemp;
    Prob->EucGrad(StieX, &StieGrad);
    double GnormSq = ProjectGradAndReturnNormSq(StieX, &StieGrad, symUtV, n, p);
    
    int it = 0;
    while (sqrt(GnormSq) > tol && step_size > 1e-13) {
        retract(StieX, &StieGrad, &StieXtemp, -1.0 * step_size, YV, VY, YVVY, VYY, pivots, n, p);
        FXtemp = Prob->f(&StieXtemp);
        
        while (FXtemp < FX - 0.7 * step_size * GnormSq) {
            step_size *= alpha;
            retract(StieX, &StieGrad, &StieXtemp, -1.0 * step_size, YV, VY, YVVY, VYY, pivots, n, p);
            FXtemp = Prob->f(&StieXtemp);
        }
        
        while (FXtemp > FX - 0.5 * step_size * GnormSq && step_size > 1e-13) {
            step_size /= alpha;
            retract(StieX, &StieGrad, &StieXtemp, -1.0 * step_size, YV, VY, YVVY, VYY, pivots, n, p);
            FXtemp = Prob->f(&StieXtemp);
        }
        StieCopy(&StieXtemp, StieX, n, p);
        FX = Prob->f(StieX);
        Prob->EucGrad(StieX, &StieGrad);
        GnormSq = ProjectGradAndReturnNormSq(StieX, &StieGrad, symUtV, n, p);
        ++it;
    }
    
    delete[] symUtV;
    delete[] VY;
    delete[] YV;
    delete[] YVVY;
    delete[] VYY;
    delete[] pivots;
    
    return it;
}

int AcceleratedGradDescentGradRestart(StieVariable* StieX, Problem* Prob, double tol, integer n, integer p, integer* nf, integer* ng) {
    // Parameters controlling when we restart and by which factor we decrease the
    // step size during the armijo search.
    const double alpha = 1.7;
    
    // Allocate temp memory for future calculations.
    
    double *symUtV = new double[p * p];
    double *invMat = new double[p * p];
    double *UTV = new double[p * p];
    double *VY = new double[n * p * 2];
    double *YV = new double[n * p * 2];
    double *YVVY = new double[4 * p * p];
    double *VYY = new double[2 * p * p];
    integer* pivots = new integer[2 * p];
    StieVariable StieY(n,p);
    StieVariable StieXtemp(n,p);
    StieVector StieGrad(n,p);
    StieVector YtoX(n,p);
    
    // Set the initial Y and Xtemp iterate equal to X.
    StieCopy(StieX, &StieY, n, p);
    StieCopy(StieX, &StieXtemp, n, p);
    
    double step_size = 0.1;
    double FY = Prob->f(&StieY); ++(*nf);
    double FX = FY;
    Prob->EucGrad(&StieY, &StieGrad); ++(*ng);
    double GnormSq = ProjectGradAndReturnNormSq(&StieY, &StieGrad, symUtV, n, p);
    
    int it = 0;
    int n_step = 0;
    while (sqrt(GnormSq) > tol && step_size > 1e-13) {
        retract(&StieY, &StieGrad, StieX, -1.0 * step_size, YV, VY, YVVY, VYY, pivots, n, p);
        FX = Prob->f(StieX); ++(*nf);
        
        while (FX < FY - 0.7 * step_size * GnormSq) {
            step_size *= alpha;
            retract(&StieY, &StieGrad, StieX, -1.0 * step_size, YV, VY, YVVY, VYY, pivots, n, p);
            FX = Prob->f(StieX); ++(*nf);
        }           
             
        while (FX > FY - 0.5 * step_size * GnormSq && step_size > 1e-13) {
            step_size /= alpha;
            retract(&StieY, &StieGrad, StieX, -1.0 * step_size, YV, VY, YVVY, VYY, pivots, n, p);
            FX = Prob->f(StieX); ++(*nf);
        }
        find_momentum_vector(&StieXtemp, StieX, &StieGrad, symUtV, invMat, pivots, n, p);
        retract(&StieXtemp, &StieGrad, &StieY, 1.0 + n_step / (n_step + 3.0), YV, VY, YVVY, VYY, pivots, n, p);
        FY = Prob->f(&StieY); ++(*nf);
        Prob->EucGrad(&StieY, &StieGrad); ++(*ng);
        GnormSq = ProjectGradAndReturnNormSq(&StieY, &StieGrad, symUtV, n, p);
        find_momentum_vector(&StieY, StieX, &YtoX, symUtV, invMat, pivots, n, p);
        if (calculate_stie_inner_product(&YtoX, &StieGrad, &StieY, symUtV, invMat, UTV, n, p) < -1.0 * step_size * GnormSq) {
            n_step = 0;
            StieCopy(&StieXtemp, StieX, n, p);
            StieCopy(&StieXtemp, &StieY, n, p);
            FY = Prob->f(&StieY); ++(*nf);
            Prob->EucGrad(&StieY, &StieGrad); ++(*ng);
            GnormSq = ProjectGradAndReturnNormSq(&StieY, &StieGrad, symUtV, n, p);
        } else {
            StieCopy(StieX, &StieXtemp, n, p);
            ++n_step;
        }
        ++it;
    }
    
    printf("%d %lf\n", it, sqrt(GnormSq));
    
    delete[] symUtV;
    delete[] UTV;
    delete[] VY;
    delete[] invMat;
    delete[] YV;
    delete[] YVVY;
    delete[] VYY;
    delete[] pivots;
    
    return it;
}

int AcceleratedGradDescentFunctionRestart(StieVariable* StieX, Problem* Prob, double tol, integer n, integer p, integer* nf, integer* ng) {
    // Parameters controlling when we restart and by which factor we decrease the
    // step size during the armijo search.
    const double r_rho = 0.01;
    const double alpha = 1.7;
    
    // Allocate temp memory for future calculations.
    
    double *symUtV = new double[p * p];
    double *invMat = new double[p * p];
    double *VY = new double[n * p * 2];
    double *YV = new double[n * p * 2];
    double *YVVY = new double[4 * p * p];
    double *VYY = new double[2 * p * p];
    integer* pivots = new integer[2 * p];
    StieVariable StieY(n,p);
    StieVariable StieXtemp(n,p);
    StieVector StieGrad(n,p);
    
    // Set the initial Y and Xtemp iterate equal to X.
    StieCopy(StieX, &StieY, n, p);
    StieCopy(StieX, &StieXtemp, n, p);
    
    double step_size = 0.1;
    double FY = Prob->f(&StieY); ++(*nf);
    double FX = FY;
    double FXtemp = FY;
    Prob->EucGrad(&StieY, &StieGrad); ++(*ng);
    double GnormSq = ProjectGradAndReturnNormSq(&StieY, &StieGrad, symUtV, n, p);
    
    int it = 0;
    int n_step = 0;
    while (sqrt(GnormSq) > tol && step_size > 1e-13) {
        retract(&StieY, &StieGrad, StieX, -1.0 * step_size, YV, VY, YVVY, VYY, pivots, n, p);
        FX = Prob->f(StieX); ++(*nf);
        
        while (FX < FY - 0.9 * step_size * GnormSq) {
            step_size *= alpha;
            retract(&StieY, &StieGrad, StieX, -1.0 * step_size, YV, VY, YVVY, VYY, pivots, n, p);
            FX = Prob->f(StieX); ++(*nf);
        }           
             
        while (FX > FY - 0.5 * step_size * GnormSq && step_size > 1e-13) {
            step_size /= alpha;
            retract(&StieY, &StieGrad, StieX, -1.0 * step_size, YV, VY, YVVY, VYY, pivots, n, p);
            FX = Prob->f(StieX); ++(*nf);
        }
        if (FX > FXtemp - r_rho*step_size*GnormSq) {
            n_step = 0;
            StieCopy(&StieXtemp, StieX, n, p);
            StieCopy(&StieXtemp, &StieY, n, p);
        } else {
            FXtemp = FX;
            find_momentum_vector(&StieXtemp, StieX, &StieGrad, symUtV, invMat, pivots, n, p);
            retract(&StieXtemp, &StieGrad, &StieY, 1.0 + n_step / (n_step + 3.0), YV, VY, YVVY, VYY, pivots, n, p);
            StieCopy(StieX, &StieXtemp, n, p);
            ++n_step;
        }
        FY = Prob->f(&StieY); ++(*nf);
        Prob->EucGrad(&StieY, &StieGrad); ++(*ng);
        GnormSq = ProjectGradAndReturnNormSq(&StieY, &StieGrad, symUtV, n, p);
        ++it;
    }
    
    printf("%d %lf\n", it, sqrt(GnormSq));
    
    delete[] symUtV;
    delete[] VY;
    delete[] invMat;
    delete[] YV;
    delete[] YVVY;
    delete[] VYY;
    delete[] pivots;
    
    return it;
}

/*This function checks the number and formats of input parameters.
nlhs: the number of output in mxArray format
plhs: the output objects in mxArray format
nrhs: the number of input in mxArray format
prhs: the input objects in mxArray format */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nlhs != 5) mexErrMsgTxt("The number of outputs should be four.\n");
    if (nrhs != 5) mexErrMsgTxt("The number of inputs should be five.\n");
    
    // Read input into C++ format. Note that B is a sparse matrix
    double *B, *D, *X;
    double tol;
	B = mxGetPr(prhs[0]);
	D = mxGetPr(prhs[1]);
	X = mxGetPr(prhs[2]);
    tol = mxGetScalar(prhs[3]);
    int flag = (int) *mxGetPr(prhs[4]);
    
    /* dimensions of input matrices */
    integer p, n, nzmax;
    size_t *inir, *injc;
    
	nzmax = mxGetNzmax(prhs[0]);
	inir = mxGetIr(prhs[0]);
	injc = mxGetJc(prhs[0]);
	n = mxGetM(prhs[0]);
	p = mxGetM(prhs[1]);
	unsigned long long *ir = new unsigned long long[nzmax + n + 1];
	unsigned long long *jc = ir + nzmax;
	for (integer i = 0; i < nzmax; i++)
		ir[i] = inir[i];
	for (integer i = 0; i < n + 1; i++)
		jc[i] = injc[i];

	/*Check the correctness of the inputs*/
	if (mxGetN(prhs[0]) != n)
	{
		mexErrMsgTxt("The size of matrix B is not correct.\n");
	}
	if (mxGetN(prhs[1]) != 1)
	{
		mexErrMsgTxt("The size of the D is not correct!\n");
	}
	if (mxGetM(prhs[2]) != n || mxGetN(prhs[2]) != p)
	{
		mexErrMsgTxt("The size of the initial X is not correct!\n");
	}
    
    printf("(n, p):%d,%d\n", n, p);
    
	// Define the Brockett problem
	Stiefel Domain(n, p);
	StieSparseBrockett Prob(B, ir, jc, nzmax, D, n, p);
	Prob.SetDomain(&Domain);
    Prob.SetUseGrad(true);
    
    Domain.SetHasHHR(0);
    
    // Move X into two StieVariables.
	StieVariable StieX(n, p);
	double *StieXptr = StieX.ObtainWriteEntireData();
	for (integer i = 0; i < n * p; i++) {
		StieXptr[i] = X[i];
    }
    
    StieVector Grad(n,p);
    Prob.f(&StieX);
    Prob.EucGrad(&StieX, &Grad);
    double* symMat = new double[p*p];
    double initial_norm = sqrt(ProjectGradAndReturnNormSq(&StieX, &Grad, symMat, n, p));
    delete[] symMat;
    
    printf("Initial Norm: %lf \n", initial_norm);
    
    int it = 0;
    integer nf = 0;
    integer ng = 0;
    clock_t begin = getTickCount();
    if (flag == 0) {
        it = GradDescent(&StieX, &Prob, initial_norm * tol, n, p);
    } else if (flag == 1) {
        it = AcceleratedGradDescentFunctionRestart(&StieX, &Prob, initial_norm * tol, n, p, &nf, &ng);
    } else if (flag == 2) {
        it = AcceleratedGradDescentGradRestart(&StieX, &Prob, initial_norm * tol, n, p, &nf, &ng);
    } else printf("The flag passed should be either 1, 2, or 3.\n");
    clock_t end = getTickCount();
    // output <- StieX
    mexProblem::ObtainMxArrayFromElement(plhs[0], &StieX);
        
    plhs[1] = mxCreateDoubleScalar((double) it);
    plhs[2] = mxCreateDoubleScalar((double) nf);
    plhs[3] = mxCreateDoubleScalar((double) ng);
    plhs[4] = mxCreateDoubleScalar((double) (end - begin) / CLK_PS);
}
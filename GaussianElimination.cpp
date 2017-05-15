//
//  main.cpp
//  GaussianElimination
//
//  Created by Milind Chabbi on 5/14/17.
//  Copyright © 2017 Milind Chabbi. All rights reserved.
//

#include <iostream>
#define _XOPEN_SOURCE
#include <stdlib.h>
#include <vector>
#include <math.h>       /* fabs */
#include <assert.h>       /* fabs */

using namespace std;

// Given a upper triangular matrix, we back substitute to find the unknown x.
void BackSubstitute(vector< vector<double> >  & A, vector<double>  & x, vector<double>  & b, int dimension){
    const int maxRows = dimension;
    const int maxCols = dimension;
    
    // last row has only one unknown, hence we start from bottom up
    for(int i = maxRows-1; i >= 0 ; i--){
        double rhs = b[i];
        // solve for x[i]
        for(int j = maxCols - 1; j > i; j--){
            rhs -= A[i][j] * x[j];
        }
        x[i] = rhs / A[i][i];
    }
}

int GetMaxRow(vector< vector<double> >  & A, int piv, int dimension){
    int newPiv = piv;
    for (int i = piv; i < dimension; i ++) {
        if (fabs(A[i][piv]) > fabs(A[newPiv][piv])){
            newPiv = i;
        }
    }
    return  newPiv;
}

// Find L2 norm
double ComputeL2(vector< vector<double> >  & A, vector<double> & x, vector<double> & b, int dimension){
    int maxRows = dimension;
    int maxCols = dimension;
    vector<double>  l2Vector(dimension);
    
    // Compute Ax - b
    for(int i = 0 ; i < maxRows ; i++){
        l2Vector[i] = 0;
        for(int j = 0 ; j < maxCols ; j++){
            l2Vector[i] += A[i][j] * x[j];
        }
        l2Vector[i] -= b[i];
    }
    
    // Find sum of square of each element
    
    double res = 0;
    for(int i = 0 ; i < maxRows ; i++){
        res += l2Vector[i] * l2Vector[i];
    }
    // Find Square root
    return sqrt(res);
}

// Gaussian elimination with partial piviting is a method to solve a system of linear equations.
// A matlab reference implementation is here: https://www.mathworks.com/matlabcentral/fileexchange/26774-gaussian-elimination-with-partial-pivoting?focused=5147355&tab=function
// The code below is a C++ single-threaded implementation.
// The system solves Ax=b, where A is a n * n matrix, x and b are n-dimensional vectors.
// The solve phase converts A into an upper triangular matrix.
// Partial pivoting is used for numerical stability.
// The back substitution phase finds the unknown(s) x.
// The solution can be verified by computing the L2norm, whose value should be close to 0.
// Your task is to:
// 1. Make this code multi-threaded using OpenMP
// 2. Make this code multi-process using MPI.
// 3. Make this code multi-threaded and multi-process via OpenMP within a node and multi-process across nodes.
// 4. Compute parallel scaling and efficiency at various thread/process counts and compare and contrast different implementations.
// Use 8000x8000 as a reference matrix for your efficiency computation.

int main(int argc, const char * argv[]) {
    // Serial case will not need workingThreads
    // Otherwise accept nthreads from commandline
#ifdef TEST
    argc = 2;
    argv[1] = "1000";
#endif
    if(argc < 2){
        printf("Usage : ./%s <dimension>", argv[0]);
        return -1;
    }
    int dimension = atoi(argv[1]);
    if (dimension <= 0){
        printf("\n You should pass dimension > 0 for me to do anything useful");
        return -1;
    }
    
    // Solve Ax = b
    
    // Allocate  matrix A[dimension][dimension]
    vector< vector<double> > A(dimension);
    for(int i = 0 ; i < dimension; i++){
        A[i].resize(dimension);
    }
    
    // Allocate vector b[dimension]
    vector<double> b(dimension);
    // Allocate vector x[dimension]
    vector<double> x(dimension);
    
    // Init rand seed
    // Use this for multi-threaded
    //struct drand48_data buffer;
    //srand48_r(0, &buffer);
    srand(0);
    
    // Init A, b
    for(int i = 0; i < dimension; i++){
        for (int j = 0; j<dimension; j++) {
            // Not right for multi-threaded code
            A[i][j] = rand();
        }
        b[i] = rand();
    }
    
    
    for(int piv = 0 ; piv < dimension; piv ++) {
        // find pivot row
        int pivotRow = GetMaxRow(A, piv, dimension);
        // swap
        swap(A[piv], A[pivotRow]);
        swap(b[piv], b[pivotRow]);
        
        // Solve
        for(int i = piv+1 ; i < dimension; i++) {
            auto scale = A[i][piv] / A[piv][piv];
            for(int j = piv; j < dimension; j++){
                A[i][j] =  A[i][j] - scale * A[piv][j];
            }
            b[i] = b[i] - scale * b[piv];
        }
    }
    
    // Back substitute
    BackSubstitute(A, x, b, dimension);
    
    // Validate; l2 norm should be close to 0.
    double l2 = ComputeL2(A, x, b, dimension);
    cout<< "L2 = " << l2;
    assert (l2 < 0.01);
    return 0;
}


//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file GenerateProblem.cpp

 HPCG routine
 */

#ifndef HPCG_NOMPI
#include <mpi.h>
#endif

#ifndef HPCG_NOOPENMP
#include <omp.h>
#endif

#if defined(HPCG_DEBUG) || defined(HPCG_DETAILED_DEBUG)
#include <fstream>
using std::endl;
#include "hpcg.hpp"
#endif
#include <cassert>

#include <stdio.h>
#include <string.h>

#include "GenerateProblem.hpp"


/*!
  Routine to read a sparse matrix, right hand side, initial guess, and exact
  solution (as computed by a direct solver).

  @param[in]  geom   data structure that stores the parallel run parameters and the factoring of total number of processes into three dimensional grid
  @param[in]  A      The known system matrix
  @param[inout] b      The newly allocated and generated right hand side vector (if b!=0 on entry)
  @param[inout] x      The newly allocated solution vector with entries set to 0.0 (if x!=0 on entry)
  @param[inout] xexact The newly allocated solution vector with entries set to the exact solution (if the xexact!=0 non-zero on entry)

  @see GenerateGeometry
*/

void GenerateProblem(SparseMatrix & A, Vector * b, Vector * x, Vector * xexact) {

  // Make local copies of geometry information.  Use global_int_t since the RHS products in the calculations
  // below may result in global range values.
  global_int_t nx = A.geom->nx;
  global_int_t ny = A.geom->ny;
  global_int_t nz = A.geom->nz;
  global_int_t npx = A.geom->npx;
  global_int_t npy = A.geom->npy;
  global_int_t npz = A.geom->npz;
  global_int_t ipx = A.geom->ipx;
  global_int_t ipy = A.geom->ipy;
  global_int_t ipz = A.geom->ipz;
  global_int_t gnx = nx*npx;
  global_int_t gny = ny*npy;
  global_int_t gnz = nz*npz;

  local_int_t localNumberOfRows = nx*ny*nz; // This is the size of our subblock
  // If this assert fails, it most likely means that the local_int_t is set to int and should be set to long long
  assert(localNumberOfRows>0); // Throw an exception of the number of rows is less than zero (can happen if int overflow)
  local_int_t numberOfNonzerosPerRow = 27; // We are approximating a 27-point finite element/volume/difference 3D stencil

  global_int_t totalNumberOfRows = ((global_int_t) localNumberOfRows)*((global_int_t) A.geom->size); // Total number of grid points in mesh
  // If this assert fails, it most likely means that the global_int_t is set to int and should be set to long long
  assert(totalNumberOfRows>0); // Throw an exception of the number of rows is less than zero (can happen if int overflow)


  // Allocate arrays that are of length localNumberOfRows
  int * nonzerosInRow = new int[localNumberOfRows];
  global_int_t ** mtxIndG = new global_int_t*[localNumberOfRows];
  local_int_t  ** mtxIndL = new local_int_t*[localNumberOfRows];
  double ** matrixValues = new double*[localNumberOfRows];
  double ** matrixDiagonal = new double*[localNumberOfRows];

  if (b!=0) InitializeVector(*b, localNumberOfRows);
  if (x!=0) InitializeVector(*x, localNumberOfRows);
  if (xexact!=0) InitializeVector(*xexact, localNumberOfRows);
  double * bv = 0;
  double * xv = 0;
  double * xexactv = 0;
  if (b!=0) bv = b->values; // Only compute exact solution if requested
  if (x!=0) xv = x->values; // Only compute exact solution if requested
  if (xexact!=0) xexactv = xexact->values; // Only compute exact solution if requested
  A.localToGlobalMap.resize(localNumberOfRows);

  // Use a parallel loop to do initial assignment:
  // distributes the physical placement of arrays of pointers across the memory system
#ifndef HPCG_NOOPENMP
  #pragma omp parallel for
#endif
  for (local_int_t i=0; i< localNumberOfRows; ++i) {
    matrixValues[i] = 0;
    matrixDiagonal[i] = 0;
    mtxIndG[i] = 0;
    mtxIndL[i] = 0;
  }
  // Now allocate the arrays pointed to
  for (local_int_t i=0; i< localNumberOfRows; ++i) {
    mtxIndL[i] = new local_int_t[numberOfNonzerosPerRow];
    matrixValues[i] = new double[numberOfNonzerosPerRow];
    mtxIndG[i] = new global_int_t[numberOfNonzerosPerRow];
  }



  local_int_t localNumberOfNonzeros = 0;
  // TODO:  This triply nested loop could be flattened or use nested parallelism
#ifndef HPCG_NOOPENMP
  #pragma omp parallel for
#endif
  for (local_int_t iz=0; iz<nz; iz++) {
    global_int_t giz = ipz*nz+iz;
    for (local_int_t iy=0; iy<ny; iy++) {
      global_int_t giy = ipy*ny+iy;
      for (local_int_t ix=0; ix<nx; ix++) {
        global_int_t gix = ipx*nx+ix;
        local_int_t currentLocalRow = iz*nx*ny+iy*nx+ix;
        global_int_t currentGlobalRow = giz*gnx*gny+giy*gnx+gix;
#ifndef HPCG_NOOPENMP
// C++ std::map is not threadsafe for writing
        #pragma omp critical
#endif
        A.globalToLocalMap[currentGlobalRow] = currentLocalRow;

        A.localToGlobalMap[currentLocalRow] = currentGlobalRow;
#ifdef HPCG_DETAILED_DEBUG
        HPCG_fout << " rank, globalRow, localRow = " << A.geom->rank << " " << currentGlobalRow << " " << A.globalToLocalMap[currentGlobalRow] << endl;
#endif
        char numberOfNonzerosInRow = 0;
        double * currentValuePointer = matrixValues[currentLocalRow]; // Pointer to current value in current row
        global_int_t * currentIndexPointerG = mtxIndG[currentLocalRow]; // Pointer to current index in current row
        for (int sz=-1; sz<=1; sz++) {
          if (giz+sz>-1 && giz+sz<gnz) {
            for (int sy=-1; sy<=1; sy++) {
              if (giy+sy>-1 && giy+sy<gny) {
                for (int sx=-1; sx<=1; sx++) {
                  if (gix+sx>-1 && gix+sx<gnx) {
                    global_int_t curcol = currentGlobalRow+sz*gnx*gny+sy*gnx+sx;
                    if (curcol==currentGlobalRow) {
                      matrixDiagonal[currentLocalRow] = currentValuePointer;
                      *currentValuePointer++ = 26.0;
                    } else {
                      *currentValuePointer++ = -1.0;
                    }
                    *currentIndexPointerG++ = curcol;
                    numberOfNonzerosInRow++;
                  } // end x bounds test
                } // end sx loop
              } // end y bounds test
            } // end sy loop
          } // end z bounds test
        } // end sz loop
        nonzerosInRow[currentLocalRow] = numberOfNonzerosInRow;
#ifndef HPCG_NOOPENMP
        #pragma omp critical
#endif
        localNumberOfNonzeros += numberOfNonzerosInRow; // Protect this with an atomic
        if (b!=0)      bv[currentLocalRow] = 26.0 - ((double) (numberOfNonzerosInRow-1));
        if (x!=0)      xv[currentLocalRow] = 0.0;
        if (xexact!=0) xexactv[currentLocalRow] = 1.0;
      } // end ix loop
    } // end iy loop
  } // end iz loop
#ifdef HPCG_DETAILED_DEBUG
  HPCG_fout     << "Process " << A.geom->rank << " of " << A.geom->size <<" has " << localNumberOfRows    << " rows."     << endl
      << "Process " << A.geom->rank << " of " << A.geom->size <<" has " << localNumberOfNonzeros<< " nonzeros." <<endl;
#endif

  global_int_t totalNumberOfNonzeros = 0;
#ifndef HPCG_NOMPI
  // Use MPI's reduce function to sum all nonzeros
#ifdef HPCG_NO_LONG_LONG
  MPI_Allreduce(&localNumberOfNonzeros, &totalNumberOfNonzeros, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#else
  long long lnnz = localNumberOfNonzeros, gnnz = 0; // convert to 64 bit for MPI call
  MPI_Allreduce(&lnnz, &gnnz, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
  totalNumberOfNonzeros = gnnz; // Copy back
#endif
#else
  totalNumberOfNonzeros = localNumberOfNonzeros;
#endif
  // If this assert fails, it most likely means that the global_int_t is set to int and should be set to long long
  // This assert is usually the first to fail as problem size increases beyond the 32-bit integer range.
  assert(totalNumberOfNonzeros>0); // Throw an exception of the number of nonzeros is less than zero (can happen if int overflow)

  A.title = 0;
  A.totalNumberOfRows = totalNumberOfRows;
  A.totalNumberOfNonzeros = totalNumberOfNonzeros;
  A.localNumberOfRows = localNumberOfRows;
  A.localNumberOfColumns = localNumberOfRows;
  A.localNumberOfNonzeros = localNumberOfNonzeros;
  A.nonzerosInRow = nonzerosInRow;
  A.mtxIndG = mtxIndG;
  A.mtxIndL = mtxIndL;
  A.matrixValues = matrixValues;
  A.matrixDiagonal = matrixDiagonal;

  return;
}


void GenerateProblem_file(SparseMatrix & A, Vector * b, Vector * x, Vector * xexact, char * filename) {

  // Make local copies of geometry information.  Use global_int_t since the RHS products in the calculations
  // below may result in global range values.
  FILE *f;
  char name[80];
  strcpy(name, filename);
  strcat(name, ".mtx");
  f = fopen(name, "r");
  assert(f != NULL);
  local_int_t m, n, nz, base;

  fscanf(f, "%d %d %d", &m, &n, &nz);    
  base = 1;
 
  // Symmetric matrix is assumed.
  // first read all the data in.
  double * a = new double[nz * 2];
  int *    i_idx = new int[nz * 2];  
  int *    j_idx = new int[nz * 2];
  int k = 0;
  for (int i=0; i<nz; i++)  {
    fscanf(f, "%d %d %lg", &i_idx[i], &j_idx[i], &a[i]);
    i_idx[i] -= base;  /* adjust from 1-based to 0-based */
    j_idx[i] -= base;
	assert(i_idx[i] >= j_idx[i]);
	if ( i_idx[i] != j_idx[i] ){
	  i_idx[nz+k] = j_idx[i];
	  j_idx[nz+k] = i_idx[i];
	  a[nz+k] = a[i];
	  k++;
    }
  }
  int non_diag = k;
  nz += k;

  A._values = new double[nz];
  A._mtxIndG = new global_int_t[nz];
  A._mtxIndL = new local_int_t[nz];
  // rearrange the data into CSR format.
  int * row_start = new int[m+1]; 
  for (int i=0; i<=m; i++) row_start[i] = 0;

  
  for (int i=0; i<nz; i++) row_start[i_idx[i]+1]++;

  for (int i=0; i<n; i++) row_start[i+1] += row_start[i];
  assert(row_start[m] == nz);
  
  local_int_t localNumberOfRows = m;
  double ** matrixDiagonal = new double*[localNumberOfRows];
  int * nonzerosInRow = new int[localNumberOfRows];
  global_int_t ** mtxIndG = new global_int_t*[localNumberOfRows];
  local_int_t  ** mtxIndL = new local_int_t*[localNumberOfRows];
  double ** matrixValues = new double*[localNumberOfRows];

  for (local_int_t i=0; i< localNumberOfRows; ++i) {
    matrixValues[i] = 0;
    matrixDiagonal[i] = 0;
    mtxIndG[i] = 0;
    mtxIndL[i] = 0;
  }

  // sort data in each row in the following order. 
  // lower triangle -> diagonal -> upper triangle.
  for (int i = nz - non_diag; i < nz; i++) {
   	int r = i_idx[i];
	int c = j_idx[i];
	assert( c > r);
    A._values[ row_start[r] ] = a[i];
	A._mtxIndG[ row_start[r] ] = c; 
	A._mtxIndL[ row_start[r] ] = c; 
    row_start[r] ++;
  }
  for (int i = 0; i < nz - non_diag; i++) {
	int r = i_idx[i];
	int c = j_idx[i];
	int idx = row_start[r];
    A._values[ idx ] = a[i];
	A._mtxIndG[ idx ] = c; 
	A._mtxIndL[ idx ] = c; 
	assert(c <= r);
	if (r == c)
      matrixDiagonal[r] = &A._values[ row_start[r] ];
    row_start[r] ++;
  }
  assert(row_start[m - 1] == nz);

  // rollback row_start.
  for (int i = m; i > 0; i--)
	  row_start[i] = row_start[i-1];
  row_start[0] = 0;

  delete a;

  // translate the CSR format to the SparseMatrix format.

  assert(localNumberOfRows>0); // Throw an exception of the number of rows is less than zero (can happen if int overflow)

  global_int_t totalNumberOfRows = m;
  assert(totalNumberOfRows>0); // Throw an exception of the number of rows is less than zero (can happen if int overflow)
    
  if (b!=0) InitializeVector(*b, localNumberOfRows);
  if (x!=0) InitializeVector(*x, localNumberOfRows);
  if (xexact!=0) InitializeVector(*xexact, localNumberOfRows);
  double * bv = 0;
  double * xv = 0;
  double * xexactv = 0;
  if (b!=0) bv = b->values; // Only compute exact solution if requested
  if (x!=0) xv = x->values; // Only compute exact solution if requested
  if (xexact!=0) xexactv = xexact->values; // Only compute exact solution if requested
  A.localToGlobalMap.resize(localNumberOfRows);
  // Use a parallel loop to do initial assignment:
  // distributes the physical placement of arrays of pointers across the memory system
#ifndef HPCG_NOOPENMP
  #pragma omp parallel for
#endif
    // Now allocate the arrays pointed to
  for (local_int_t i=0; i< localNumberOfRows; ++i) {
	int nzInRow = row_start[i + 1] - row_start[i];
	nonzerosInRow[i] = nzInRow;
    mtxIndL[i] = &A._mtxIndL[ row_start[i] ];
    matrixValues[i] = &A._values[ row_start[i] ]; 
    mtxIndG[i] = &A._mtxIndG[ row_start[i] ];
  }

  for (int row = 0; row < localNumberOfRows; row ++) {
	bool divide_find = false;
	int divide_idx = 0;
	for (int idx = 0; idx < nonzerosInRow[row]; idx ++) {
	  if (mtxIndL[row][idx] == row && !divide_find)
		continue;
	  if (mtxIndL[row][idx] < row && !divide_find) {
		divide_find = true;
	    divide_idx = idx;
	  }
	  if (mtxIndL[row][idx] == row) {
		local_int_t tmpL = mtxIndL[row][idx];
		mtxIndL[row][idx] = mtxIndL[row][divide_idx];
		mtxIndL[row][divide_idx] = tmpL;

		global_int_t tmpG = mtxIndG[row][idx];
		mtxIndG[row][idx] = mtxIndG[row][divide_idx];
		mtxIndG[row][divide_idx] = tmpG;
		
		double tmp = matrixValues[row][idx];
		matrixValues[row][idx] = matrixValues[row][divide_idx];
		matrixValues[row][divide_idx] = tmp;
        matrixDiagonal[row] = &matrixValues[row][divide_idx]; 
	  }
	}
  }
  
  for (int row = 0; row < localNumberOfRows; row ++) {
	bool diag = false;
	for (int idx = 0; idx < nonzerosInRow[row]; idx ++) {
	  if (mtxIndL[row][idx] == row) {
		diag = true;
		matrixDiagonal[row] = matrixValues[row];
	  }
      if (!diag) 	assert(mtxIndL[row][idx] > row);
	  else 			assert(mtxIndL[row][idx] <= row);
	}
	assert(diag);
  }
  global_int_t totalNumberOfNonzeros = 0;
  totalNumberOfNonzeros = nz;
  // If this assert fails, it most likely means that the global_int_t is set to int and should be set to long long
  // This assert is usually the first to fail as problem size increases beyond the 32-bit integer range.
  assert(totalNumberOfNonzeros>0); // Throw an exception of the number of nonzeros is less than zero (can happen if int overflow)

  A.title = 0;
  A.totalNumberOfRows = totalNumberOfRows;
  A.totalNumberOfNonzeros = totalNumberOfNonzeros;
  A.localNumberOfRows = localNumberOfRows;
  A.localNumberOfColumns = localNumberOfRows;
  A.localNumberOfNonzeros = nz;
  A.nonzerosInRow = nonzerosInRow;
  A.mtxIndG = mtxIndG;
  A.mtxIndL = mtxIndL;
  A.matrixValues = matrixValues;
  A.matrixDiagonal = matrixDiagonal;
 
  ///////////////////////////////////
  // initialize b.
  ///////////////////////////////////
  strcpy(name, filename);
  strcat(name, "_b.mtx");
  f = fopen(name, "r");
  local_int_t bm, bn; 
  fscanf(f, "%d %d", &bm, &bn);
  assert(bm == m);
  for (int i = 0; i < bm; i++)
    fscanf(f, "%lg", &b->values[i]);     /* indentifies starting index */
  
  fclose(f);
  return;
}

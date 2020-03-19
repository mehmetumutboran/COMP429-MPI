 /*
 * Solves the Panfilov model using an explicit numerical scheme.
 * Based on code orginally provided by Xing Cai, Simula Research
 * Laboratory and reimplementation by Scott B. Baden, UCSD
 *
 * Modified and  restructured by Didem Unat, Koc University
 *
 */
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>
#include <omp.h>
#include "mpi.h"
using namespace std;


// Utilities
// We write two printArray functions for debugging purposes.

void printArray(double* arr, int size, int myrank)
{
  int i;
  printf("My rank is %d\n",myrank);
  printf("[\n");
  for(i = 0; i < size; i++)
    printf(" %lf ", *(arr+i));
  printf("]\n");
}

void printArray(double** arr, int row, int col, int myrank, const char* msg, int info)
{
  int i, j;
  printf("%s My rank is %d\n", msg, myrank);
  printf("[\n");
  for(i = info; i < row+info; i++){
    for(j = info; j < col+info; j++)
      printf(" %lf ", arr[i][j]);
    printf("\n");
  }
  printf("]\n");
}

// Timer
// Make successive calls and take a difference to get the elapsed time.

static const double kMicro = 1.0e-6;
double getTime()
{
    struct timeval TV;
    struct timezone TZ;

    const int RC = gettimeofday(&TV, &TZ);
    if(RC == -1) {
            cerr << "ERROR: Bad call to gettimeofday" << endl;
            return(-1);
    }

    return( ((double)TV.tv_sec) + kMicro * ((double)TV.tv_usec) );

}  // end getTime()

// Allocate a 2D array
double **alloc2D(int m,int n){
   double **E;
   int nx=n, ny=m;
   E = (double**)malloc(sizeof(double*)*ny + sizeof(double)*nx*ny);
   assert(E);
   int j;
   for(j=0;j<ny;j++)
     E[j] = (double*)(E+ny) + j*nx;
   return(E);
}

// Reports statistics about the computation
// These values should not vary (except to within roundoff)
// when we use different numbers of  processes to solve the problem
 double stats(double **E, int m, int n, double *_mx){
     double mx = -1;
     double l2norm = 0;
     int i, j;
     for (j=1; j<=m; j++)
       for (i=1; i<=n; i++) {
	   l2norm += E[j][i]*E[j][i];
	   if (E[j][i] > mx)
	       mx = E[j][i];
      }
     *_mx = mx;
     l2norm /= (double) ((m)*(n));
     l2norm = sqrt(l2norm);
     return l2norm;
 }

// External functions
extern "C" {
    void splot(double **E, double T, int niter, int m, int n);
}
void cmdLine(int argc, char *argv[], double& T, int& n, int& px, int& py, int& plot_freq, int& no_comm, int&num_threads);


void simulate1x1 (double** E,  double** E_prev, double** R,
	       const double alpha, const int n, const int m, const double kk,
	       const double dt, const double a, const double epsilon,
	       const double M1,const double  M2, const double b, int num_threads)
{ // Simulation for 1 process works like serial version.
  int i, j;
    /*
     * Copy data from boundary of the computational box
     * to the padding region, set up for differencing
     * on the boundary of the computational box
     * Using mirror boundaries
     */
  // 1x1 n = n m = m

#pragma omp parallel if(num_threads != -1) num_threads(num_threads)
  {
     #pragma omp for
      for (j=1; j<=m; j++)
	E_prev[j][0] = E_prev[j][2];

      #pragma omp for
       for (j=1; j<=m; j++)
	E_prev[j][n+1] = E_prev[j][n-1];

      #pragma omp for
      for (i=1; i<=n; i++)
	E_prev[0][i] = E_prev[2][i];

      #pragma omp for
      for (i=1; i<=n; i++)
	E_prev[m+1][i] = E_prev[m-1][i];

    // Solve for the excitation, the PDE
    #pragma omp for collapse(2)
    for (j=1; j<=m; j++)
      for (i=1; i<=n; i++)
	E[j][i] = E_prev[j][i]+alpha*(E_prev[j][i+1]+E_prev[j][i-1]-4*E_prev[j][i]+E_prev[j+1][i]+E_prev[j-1][i]);

    /*
     * Solve the ODE, advancing excitation and recovery to the
     *     next timtestep
     */
    #pragma omp for collapse(2)
    for (j=1; j<=m; j++)
      for (i=1; i<=n; i++)
	E[j][i] = E[j][i] -dt*(kk* E[j][i]*(E[j][i] - a)*(E[j][i]-1)+ E[j][i] *R[j][i]);

    #pragma omp for collapse(2)
    for (j=1; j<=m; j++)
      for (i=1; i<=n; i++)
	R[j][i] = R[j][i] + dt*(epsilon+M1* R[j][i]/( E[j][i]+M2))*(-R[j][i]-kk* E[j][i]*(E[j][i]-b-1));

  }

}

void simulate2D (double** E,  double** E_prev, double** R,
	       const double alpha, const int n, const int m, const double kk,
	       const double dt, const double a, const double epsilon,
	       const double M1,const double  M2, const double b, int num_threads,
	       int myrank, int size, int gridsize)
{ // Simulation for 2D geometry.
  int i, j;
    /*
     * Copy data from boundary of the computational box
     * to the padding region, set up for differencing
     * on the boundary of the computational box
     * Using mirror boundaries
     */
  // In 2D Geometry n = npx, m = npy, gridsize/m = py, gridsize/n = px
  int rows = -1, cols = -1, py = -1, px = -1; // We use rows, cols to store the row and col number of the given geometry. px and py is for storing the geometry numbers.
  px = gridsize/n; py = gridsize/m;
  rows = myrank % py;
  cols = (myrank - rows) / py;

#pragma omp parallel if(num_threads != -1) num_threads(num_threads)
  {
    if(cols == 0){ // If the process is in the westernmost col of the 2D geometry it should mirror its westernmost strip.
     #pragma omp for
      for (j=1; j<=m; j++)
	E_prev[j][0] = E_prev[j][2];
    }

    if(cols == px - 1){ // If the process is in the easternmost col of the 2D geometry it should mirror its easternmost strip.
      #pragma omp for
       for (j=1; j<=m; j++)
	E_prev[j][n+1] = E_prev[j][n-1];
       }

    if(rows == 0){ // If the process is in the top row of the 2D geometry it should mirror its topmost strip.
      #pragma omp for
      for (i=1; i<=n; i++)
	E_prev[0][i] = E_prev[2][i];
      }

    if(rows == py - 1){ // If the process is in the bottom row of the 2D geometry it should mirror its bottommost strip.
      #pragma omp for
      for (i=1; i<=n; i++)
	E_prev[m+1][i] = E_prev[m-1][i];
      }

    // Solve for the excitation, the PDE
    #pragma omp for collapse(2)
    for (j=1; j<=m; j++)
      for (i=1; i<=n; i++)
	E[j][i] = E_prev[j][i]+alpha*(E_prev[j][i+1]+E_prev[j][i-1]-4*E_prev[j][i]+E_prev[j+1][i]+E_prev[j-1][i]);


    /*
     * Solve the ODE, advancing excitation and recovery to the
     *     next timtestep
     */
    #pragma omp for collapse(2)
    for (j=1; j<=m; j++)
      for (i=1; i<=n; i++)
	E[j][i] = E[j][i] -dt*(kk* E[j][i]*(E[j][i] - a)*(E[j][i]-1)+ E[j][i] *R[j][i]);

    #pragma omp for collapse(2)
    for (j=1; j<=m; j++)
      for (i=1; i<=n; i++)
	R[j][i] = R[j][i] + dt*(epsilon+M1* R[j][i]/( E[j][i]+M2))*(-R[j][i]-kk* E[j][i]*(E[j][i]-b-1));

  }

}

void simulate1xN (double** E,  double** E_prev, double** R,
	       const double alpha, const int n, const int m, const double kk,
	       const double dt, const double a, const double epsilon,
	       const double M1,const double  M2, const double b, int num_threads,
	       int myrank, int size)
{ // Simulation for 1xn geometry. px = 1, py = n
  int i, j;
    /*
     * Copy data from boundary of the computational box
     * to the padding region, set up for differencing
     * on the boundary of the computational box
     * Using mirror boundaries
     */
  // 1xn n = m m = np

#pragma omp parallel if(num_threads != -1) num_threads(num_threads)
  {
     #pragma omp for
      for (j=1; j<=m; j++) // All the processes should mirror its westernmost strip.
	E_prev[j][0] = E_prev[j][2];

      #pragma omp for
       for (j=1; j<=m; j++) // All the processes should mirror its easternmost strip.
	E_prev[j][n+1] = E_prev[j][n-1];


    if(myrank == 0){ // The top process should mirror its topmost strip.
      #pragma omp for
      for (i=1; i<=n; i++)
	E_prev[0][i] = E_prev[2][i];
    }

    if(myrank == size - 1){ // The bottom process should mirror its bottommost strip.
      #pragma omp for
      for (i=1; i<=n; i++)
	E_prev[m+1][i] = E_prev[m-1][i];
    }

    // Solve for the excitation, the PDE
    #pragma omp for collapse(2)
    for (j=1; j<=m; j++)
      for (i=1; i<=n; i++)
	E[j][i] = E_prev[j][i]+alpha*(E_prev[j][i+1]+E_prev[j][i-1]-4*E_prev[j][i]+E_prev[j+1][i]+E_prev[j-1][i]);



    /*
     * Solve the ODE, advancing excitation and recovery to the
     *     next timtestep
     */
    #pragma omp for collapse(2)
      for (j=1; j<=m; j++)
	for (i=1; i<=n; i++)
	  E[j][i] = E[j][i] -dt*(kk* E[j][i]*(E[j][i] - a)*(E[j][i]-1)+ E[j][i] *R[j][i]);


    #pragma omp for collapse(2)
     for (j=1; j<=m; j++)
       for (i=1; i<=n; i++)
	 R[j][i] = R[j][i] + dt*(epsilon+M1* R[j][i]/( E[j][i]+M2))*(-R[j][i]-kk* E[j][i]*(E[j][i]-b-1));

  }

}


void simulateNx1 (double** E,  double** E_prev, double** R,
	       const double alpha, const int n, const int m, const double kk,
	       const double dt, const double a, const double epsilon,
	       const double M1,const double  M2, const double b, int num_threads,
	       int myrank, int size)
{ // Simulation for nx1 geometry. px = n, py = 1
  int i, j;
    /*
     * Copy data from boundary of the computational box
     * to the padding region, set up for differencing
     * on the boundary of the computational box
     * Using mirror boundaries
     */
  // nx1  n = np m = m
  //omp_set_num_threads(num_threads);
  #pragma omp parallel if(num_threads != -1) num_threads(num_threads)
  {

    if(myrank == 0){ // The leftmost process should mirror its westernmost strip.
     #pragma omp for
      for (j=1; j<=m; j++)
	E_prev[j][0] = E_prev[j][2];
    }

    if(myrank == size - 1){ // The rightmost process should mirror its easternmost strip.
      #pragma omp for
       for (j=1; j<=m; j++)
	E_prev[j][n+1] = E_prev[j][n-1];
    }

      #pragma omp for
      for (i=1; i<=n; i++) // All the processes should mirror its southernmost strip.
	E_prev[0][i] = E_prev[2][i];

      #pragma omp for
      for (i=1; i<=n; i++) // All the processes should mirror its northernmost strip.
	E_prev[m+1][i] = E_prev[m-1][i];

    // Solve for the excitation, the PDE
    #pragma omp for collapse(2)
    for (j=1; j<=m; j++)
      for (i=1; i<=n; i++)
	E[j][i] = E_prev[j][i]+alpha*(E_prev[j][i+1]+E_prev[j][i-1]-4*E_prev[j][i]+E_prev[j+1][i]+E_prev[j-1][i]);



    /*
     * Solve the ODE, advancing excitation and recovery to the
     *     next timtestep
     */
    #pragma omp for collapse(2)
    for (j=1; j<=m; j++)
      for (i=1; i<=n; i++)
	E[j][i] = E[j][i] -dt*(kk* E[j][i]*(E[j][i] - a)*(E[j][i]-1)+ E[j][i] *R[j][i]);

    #pragma omp for collapse(2)
    for (j=1; j<=m; j++)
      for (i=1; i<=n; i++)
	R[j][i] = R[j][i] + dt*(epsilon+M1* R[j][i]/( E[j][i]+M2))*(-R[j][i]-kk* E[j][i]*(E[j][i]-b-1));

  }

}



// Main program
int main (int argc, char** argv)
{
  /*
   *  Solution arrays
   *   E is the "Excitation" variable, a voltage
   *   R is the "Recovery" variable
   *   E_prev is the Excitation variable for the previous timestep,
   *      and is used in time integration
   */

  int myrank, P, np, npx, npy, rows, cols, myrowrank, mycolrank, col_size, row_size;
  double **E, **R, **E_prev, **my_E, **my_E_prev, **my_R, **gather_arr;
  double **sendbuf, **recvbuf, **sendbufpx, **sendbufpy, **recvbufpx, **recvbufpy;
  MPI_Comm row_comm, col_comm;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &P);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  // Various constants - these definitions shouldn't change
  const double a=0.1, b=0.1, kk=8.0, M1= 0.07, M2=0.3, epsilon=0.01, d=5e-5;

  double T=1000.0;
  int m=200,n=200;
  int plot_freq = 0;
  int px = 1, py = 1;
  int no_comm = 0;

  int num_threads=1;
  int next, prev, prev_row, next_row, next_col, prev_col, tag1 = 1, tag2 = 2, tag3 = 3, tag4 = 4;
  int i,j = 0, sendpx_i = 0, recvpx_i = 0, sendpy_i = 0, recvpy_i = 0, reqs_i = 0;

  MPI_Request reqs[8];
  MPI_Status stat[8];

  cmdLine( argc, argv, T, n, px, py, plot_freq, no_comm, num_threads);
  m = n;

  if(myrank == 0){
    if(px <= 0 || py <= 0){
      fprintf(stderr,"Please enter non-negative px and py!\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
      exit(1);
    }
    if(px * py != P){
      fprintf(stderr,"Please provide correct # processes (P=%d!=px*py=%d) for your px = %d and py = %d\n",P,px*py,px,py);
      MPI_Abort(MPI_COMM_WORLD, 1);
      exit(1);
    }

    if(px == 1 || py == 1){
      if(px == 1 && m%py != 0){
	fprintf(stderr,"Number of processes for y axis (py = %d) don't evenly divide grid size %d\n",py,m);
	MPI_Abort(MPI_COMM_WORLD, 1);
	exit(1);
      } else if(py == 1 && m%px != 0){
	fprintf(stderr,"Number of processes for x axis (px = %d) don't evenly divide grid size %d\n",px,m);
	MPI_Abort(MPI_COMM_WORLD, 1);
	exit(1);
      }
    } else { // 2D
      if(m%py != 0){
	fprintf(stderr,"Number of processes for y axis (py = %d) don't evenly divide grid size %d\n",py,m);
	MPI_Abort(MPI_COMM_WORLD, 1);
	exit(1);
      } else if(m%px != 0){
	fprintf(stderr,"Number of processes for x axis (px = %d) don't evenly divide grid size %d\n",px,m);
	MPI_Abort(MPI_COMM_WORLD, 1);
	exit(1);
      }
    }
  }

  // Allocate contiguous memory for solution arrays
  // The computational box is defined on [1:m+1,1:n+1]
  // We pad the arrays in order to facilitate differencing on the
  // boundaries of the computation box

  E_prev = alloc2D(m+2,n+2);
  R = alloc2D(m+2,n+2);

  // Initialization
  for (j=1; j<=m; j++)
    for (i=1; i<=n; i++)
      E_prev[j][i] = R[j][i] = 0;

  for (j=1; j<=m; j++)
    for (i=n/2+1; i<=n; i++)
      E_prev[j][i] = 1.0;

  for (j=m/2+1; j<=m; j++)
    for (i=1; i<=n; i++)
      R[j][i] = 1.0;

  if(px == 1 && py == 1){ // Single process (Serial)

    E = alloc2D(m+2,n+2);

  } else if(px == 1 || py == 1){ // 1D Geometry (1xn or nx1)

    E = alloc2D(m,n);
    np = n/P;

    sendbuf = alloc2D(2, m); // only prev and next row to send = 2 x m/1
    recvbuf = alloc2D(2, m); // only prev and next row to recv = 2 x m/1

    // Local Solution Arrays
    if(px == 1){ // 1xn
      prev = myrank - 1; // Finding the process that stay in prev row of current process
      next = myrank + 1; // Finding the process that stay in next row of current process
      if(myrank == 0) prev = py - 1; // Boundary prev calculation
      if(myrank == py - 1) next = 0; // Boundary next calculation

      my_E = alloc2D(np+2, n+2);
      my_E_prev = alloc2D(np+2, n+2);
      my_R = alloc2D(np+2, n+2);
      gather_arr = alloc2D(np, m); // for 1xn Geometry gather

      // Initialization of local arrays
      for (i = 1; i <= np; i++){
	for (j = 1; j <= n; j++){
	  my_E_prev[i][j] = E_prev[myrank*np+i][j];
	  my_R[i][j] = R[myrank*np+i][j];
	}
      }
    } else if (py == 1){ // nx1

      prev = myrank - 1; // Finding the process that stay in prev col of current process
      next = myrank + 1; // Finding the process that stay in next col of current process
      if(myrank == 0) prev = px - 1; // Boundary prev calculation
      if(myrank == px - 1) next = 0; // Boundary next calculation

      my_E = alloc2D(n+2, np+2);
      my_E_prev = alloc2D(n+2, np+2);
      my_R = alloc2D(n+2, np+2);
      gather_arr = alloc2D(m, np); // for nx1 Geometry gather

      // Initialization of local arrays
      for (i = 1; i <= n; i++){
	for (j = 1; j <= np; j++){
	  my_E_prev[i][j] = E_prev[i][myrank*np+j];
	  my_R[i][j] = R[i][myrank*np+j];
	}
      }
    }

  } else { // 2D Geometry

    E = alloc2D(m,n);

    npx = n/px; // Num elements on north and south
    npy = n/py; // Num elements on east and west

    sendbufpx = alloc2D(2, npx); // prev and next, row and col to send = 2 x npx(m/px)
    recvbufpx = alloc2D(2, npx); // prev and next, row and col to recv = 2 x npx(m/px)

    sendbufpy = alloc2D(2, npy); // prev and next, row and col to send = 2 x npy(m/py)
    recvbufpy = alloc2D(2, npy); // prev and next, row and col to recv = 2 x npy(m/py)

    cols = myrank % py;
    rows = (myrank - cols) / py;

    MPI_Comm_split(MPI_COMM_WORLD, rows, myrank, &row_comm); // We splitted the row comm world
    MPI_Comm_split(MPI_COMM_WORLD, cols, myrank, &col_comm); // We splitted the col comm world
    MPI_Comm_rank(row_comm, &myrowrank); // Find the rank in row comm world
    MPI_Comm_size(row_comm, &row_size);  // Find the size of row comm world
    MPI_Comm_rank(col_comm, &mycolrank); // Find the rank in col comm world
    MPI_Comm_size(col_comm, &col_size);  // Find the size of col comm world

    prev_row = myrowrank - 1; // Finding the process that stay in prev row of current process in row comm world
    next_row = myrowrank + 1; // Finding the process that stay in next row of current process in row comm world
    prev_col = mycolrank - 1; // Finding the process that stay in prev col of current process in col comm world
    next_col = mycolrank + 1; // Finding the process that stay in next col of current process in col comm world
    if(myrowrank == 0) prev_row = py - 1; // Boundary prev row calculation
    if(myrowrank == py - 1) next_row = 0; // Boundary next row calculation
    if(mycolrank == 0) prev_col = px - 1; // Boundary prev col calculation
    if(mycolrank == px - 1) next_col = 0; // Boundary next col calculation

    // Local Solution Arrays
    my_E = alloc2D(npy+2, npx+2);
    my_E_prev = alloc2D(npy+2, npx+2);
    my_R = alloc2D(npy+2, npx+2);
    gather_arr = alloc2D(npy, npx); // For 2D geometry gather

    // Initialization of local arrays
    for (i = 1; i <= npy; i++){
      for (j = 1; j <= npx; j++){
	my_E_prev[i][j] = E_prev[myrowrank * npy + i][mycolrank * npx + j];
	my_R[i][j] = R[myrowrank * npy + i][mycolrank * npx + j];
      }
    }
  }

  double dx = 1.0/n;

  // For time integration, these values shouldn't change
  double rp= kk*(b+1)*(b+1)/4;
  double dte=(dx*dx)/(d*4+((dx*dx))*(rp+kk));
  double dtr=1/(epsilon+((M1/M2)*rp));
  double dt = (dte<dtr) ? 0.95*dte : 0.95*dtr;
  double alpha = d*dt/(dx*dx);

  if(myrank == 0)
  {
    if(num_threads != -1) cout << P << " MPI + " << num_threads << " OpenMP" << endl;
    else cout << P << " MPI + " << "NO OpenMP" << endl;
    cout << "Grid Size       : " << n << endl;
    cout << "Duration of Sim : " << T << endl;
    cout << "Time step dt    : " << dt << endl;
    cout << "Process geometry: " << px << " x " << py << endl;
    if (no_comm)
      cout << "Communication   : DISABLED" << endl;

    cout << endl;
  }

  // Start the timer
  double t0 = getTime();

  // Simulated time is different from the integer timestep number
  // Simulated time
  double t = 0.0;
  // Integer timestep number
  int niter=0;

  // BEGINNING OF SIMULATIONS

  if(px == 1 && py == 1){ // 1 process
    while (t<T) {

      t += dt;
      niter++;

      simulate1x1(E, E_prev, R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b, num_threads);

      //swap current E with previous E
      double **tmp = E; E = E_prev; E_prev = tmp;

      /*if (plot_freq){
        int k = (int)(t/plot_freq);
        if ((t - k * plot_freq) < dt){
  	splot(E,t,niter,m+2,n+2);
        }
	}*/
    }

  }else if(px == 1){ // 1D Geometry with px = 1
    while (t < T) {

      if(no_comm){ // If MPI communications disabled(The result should be wrong)

        simulate1xN(my_E, my_E_prev, my_R, alpha, m, np, kk, dt, a, epsilon, M1, M2, b, num_threads, myrank, P);
        //swap current E with previous E
        double **tmp = my_E; my_E = my_E_prev; my_E_prev = tmp;

        /*if (plot_freq){ //edit
	  if(myrank == 0){
	    int k = (int)(t/plot_freq);
	    if ((t - k * plot_freq) < dt){
	      splot(E,t,niter,m,n);
	    }
	  }
        }*/
        t += dt;
        niter++;
        continue;
      }

      if(myrank != 0 || myrank != P - 1){ // Ordinary cell (not north or south cell)

	// Packing rows to send south and north

	memcpy(&(sendbuf[0][0]), &(my_E_prev[1][1]), sizeof(double)*m);
	memcpy(&(sendbuf[1][0]), &(my_E_prev[np][1]), sizeof(double)*m);

	MPI_Irecv(recvbuf[0], m, MPI_DOUBLE, prev, tag1, MPI_COMM_WORLD, &reqs[0]); // recv north
	MPI_Irecv(recvbuf[1], m, MPI_DOUBLE, next, tag2, MPI_COMM_WORLD, &reqs[1]); // recv south

	MPI_Isend(sendbuf[0], m, MPI_DOUBLE, prev, tag2, MPI_COMM_WORLD, &reqs[2]); // send recvs south
	MPI_Isend(sendbuf[1], m, MPI_DOUBLE, next, tag1, MPI_COMM_WORLD, &reqs[3]); // send recvs north

	t += dt;
	niter++;

	MPI_Waitall(4, reqs, stat);

      } else if(myrank == 0) { // North cell

	// Packing row to send south

	memcpy(&(sendbuf[0][0]), &(my_E_prev[np][1]), sizeof(double)*m);

	MPI_Irecv(recvbuf[0], m, MPI_DOUBLE, next, tag2, MPI_COMM_WORLD, &reqs[0]); // recv south

	MPI_Isend(sendbuf[0], m, MPI_DOUBLE, next, tag1, MPI_COMM_WORLD, &reqs[1]); // send recvs north

	t += dt;
	niter++;

	MPI_Waitall(2, reqs, stat);

      } else if(myrank == P - 1){ // South cell

	// Packing row to send north

	memcpy(&(sendbuf[0][0]), &(my_E_prev[1][1]), sizeof(double)*m);

	MPI_Irecv(recvbuf[0], m, MPI_DOUBLE, prev, tag1, MPI_COMM_WORLD, &reqs[0]); // recv north

	MPI_Isend(sendbuf[0], m, MPI_DOUBLE, prev, tag2, MPI_COMM_WORLD, &reqs[1]); // send recvs south

	t += dt;
	niter++;

	MPI_Waitall(2, reqs, stat);
      }

      // UNPACK
      if(myrank != 0 || myrank != P - 1){
	memcpy(&(my_E_prev[0][1]), &(recvbuf[0][0]), sizeof(double)*m);
	memcpy(&(my_E_prev[np+1][1]), &(recvbuf[1][0]), sizeof(double)*m);
      } else if (myrank == 0){
	memcpy(&(my_E_prev[np+1][1]), &(recvbuf[0][0]), sizeof(double)*m);
      } else if (myrank == P - 1){
	memcpy(&(my_E_prev[0][1]), &(recvbuf[0][0]), sizeof(double)*m);
      }

      simulate1xN(my_E, my_E_prev, my_R, alpha, m, np, kk, dt, a, epsilon, M1, M2, b, num_threads, myrank, P);
      //swap current E with previous E
      double **tmp = my_E; my_E = my_E_prev; my_E_prev = tmp;

   /*if (plot_freq){ //edit
	for(i = 1; i<=np; i++)
	  for(j = 1; j<=m ;j++)
	    gather_arr[i-1][j-1] = my_E[i][j];

	MPI_Gather(&(gather_arr[0][0]), m*np, MPI_DOUBLE, &(E[myrank*np][0]), m*np, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if(myrank == 0){
	  int k = (int)(t/plot_freq);
	  if ((t - k * plot_freq) < dt){
	    splot(E,t,niter,m,n);
	  }
	}
      }*/
    }// end of while loop

    if(!no_comm){
      for(i = 1; i<=np; i++) // We filled gather_arr in this loop with newly calculated values of my_E_prev
	memcpy(&(gather_arr[i-1][0]),&(my_E_prev[i][1]),sizeof(double)*m);
        /*for(j = 1; j<=m; j++)
          gather_arr[i-1][j-1] = my_E_prev[i][j];*/

      MPI_Gather(&(gather_arr[0][0]), m*np, MPI_DOUBLE, &(E[myrank*np][0]), m*np, MPI_DOUBLE, 0, MPI_COMM_WORLD); // We used gather to fill the E array with calculated new values of my_E_prev (To use in stats)
    }

  }else if(py == 1){ // 1D Geometry with py = 1
    while (t < T) {

      if(no_comm){ // If MPI communications disabled(The result should be wrong)

        simulateNx1(my_E, my_E_prev, my_R, alpha, m, np, kk, dt, a, epsilon, M1, M2, b, num_threads, myrank, P);
        //swap current E with previous E
        double **tmp = my_E; my_E = my_E_prev; my_E_prev = tmp;

        /*if (plot_freq){ //edit
	    if(myrank == 0){
	       int k = (int)(t/plot_freq);
	          if ((t - k * plot_freq) < dt){
		     splot(E,t,niter,m,n);
		  }
	     }
	  }*/
        t += dt;
        niter++;
        continue;
      }

	if(myrank != 0 || myrank != P - 1){ // Ordinary cell (not west or east cell)

	  // Packing cols to send west and east
	  for (i = 1; i <= m; i++){
	    sendbuf[0][i-1] = my_E_prev[i][1];
	    sendbuf[1][i-1] = my_E_prev[i][np];
	  }

	  MPI_Irecv(recvbuf[0], m, MPI_DOUBLE, prev, tag1, MPI_COMM_WORLD, &reqs[0]); // recv west
	  MPI_Irecv(recvbuf[1], m, MPI_DOUBLE, next, tag2, MPI_COMM_WORLD, &reqs[1]); // recv east

	  MPI_Isend(sendbuf[0], m, MPI_DOUBLE, prev, tag2, MPI_COMM_WORLD, &reqs[2]); // send recvs east
	  MPI_Isend(sendbuf[1], m, MPI_DOUBLE, next, tag1, MPI_COMM_WORLD, &reqs[3]); // send recvs west

	  t += dt;
	  niter++;

	  MPI_Waitall(4, reqs, stat);
	} else if(myrank == 0) { // West cell

	  // Packing col to send east
	  for (i = 1; i <= m; i++){
	    sendbuf[0][i-1] = my_E_prev[i][np];
	  }

	  MPI_Irecv(recvbuf[0], m, MPI_DOUBLE, next, tag2, MPI_COMM_WORLD, &reqs[0]); // recv east

	  MPI_Isend(sendbuf[0], m, MPI_DOUBLE, next, tag1, MPI_COMM_WORLD, &reqs[1]); // send recvs west

	  t += dt;
	  niter++;

	  MPI_Waitall(2, reqs, stat);
	} else if(myrank == P - 1){ // East cell

	  // Packing col to send west
	  for (i = 1; i <= m; i++){
	    sendbuf[0][i-1] = my_E_prev[i][1];
	  }

	  MPI_Irecv(recvbuf[0], m, MPI_DOUBLE, prev, tag1, MPI_COMM_WORLD, &reqs[0]); // recv west

	  MPI_Isend(sendbuf[0], m, MPI_DOUBLE, prev, tag2, MPI_COMM_WORLD, &reqs[1]); // send recvs east

	  t += dt;
	  niter++;

	  MPI_Waitall(2, reqs, stat);
	}

	// UNPACK
	if(myrank != 0 || myrank != P - 1){
	  for(i = 1; i <= m; i++){
	    my_E_prev[i][0] = recvbuf[0][i-1];
	    my_E_prev[i][np+1] = recvbuf[1][i-1];
	  }
	} else if (myrank == 0){
	  for(i = 1; i <= m; i++){
	    my_E_prev[i][np+1] = recvbuf[0][i-1];
	  }
	} else if (myrank == P - 1){
	  for(i = 1; i <= m; i++){
	    my_E_prev[i][0] = recvbuf[0][i-1];
	  }
	}

	simulateNx1(my_E, my_E_prev, my_R, alpha, np, m, kk, dt, a, epsilon, M1, M2, b, num_threads, myrank, P);
      //swap current E with previous E
	double **tmp = my_E; my_E = my_E_prev; my_E_prev = tmp;

      /*if (plot_freq){ //edit
	for(i = 1; i<=np; i++)
	  for(j = 1; j<=m ;j++)
	    gather_arr[i-1][j-1] = my_E[i][j];

	MPI_Gather(&(gather_arr[0][0]), m*np, MPI_DOUBLE, &(E[0][myrank*np]), m*np, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if(myrank == 0){
	  int k = (int)(t/plot_freq);
	  if ((t - k * plot_freq) < dt){
	    splot(E,t,niter,m,n);
	  }
	}
      }*/
    }// end of while loop

    if(!no_comm){
      for(i = 1; i<=m; i++) // We filled gather_arr in this loop with newly calculated values of my_E_prev.
	for(j = 1; j<=np; j++)
	  gather_arr[i-1][j-1] = my_E_prev[i][j];

      MPI_Gather(&(gather_arr[0][0]), m*np, MPI_DOUBLE, &(E[0][myrank*np]), m*np, MPI_DOUBLE, 0, MPI_COMM_WORLD); // We used gather to fill the E array with calculated new values of my_E_prev (To use in stats)
    }

  }else { // 2D Geometry
    while (t < T){

      if(no_comm){ // If MPI communications disabled(The result should be wrong)
	printf("No comm!\n");
        simulate2D(my_E, my_E_prev, my_R, alpha, npx, npy, kk, dt, a, epsilon, M1, M2, b, num_threads, myrank, P, m);
        //swap current E with previous E
        double **tmp = my_E; my_E = my_E_prev; my_E_prev = tmp;

        /*if (plot_freq){ //edit
         if(myrank == 0){
          int k = (int)(t/plot_freq);
           if ((t - k * plot_freq) < dt){
            splot(E,t,niter,m,n);
           }
          }
        }*/
        t += dt;
        niter++;
        continue;
      }

      //Fill sendbuffer corners, np elements
      if(myrowrank != 0 && myrowrank != py - 1 &&
	 mycolrank != 0 && mycolrank != px - 1){ // ordinary cell
	memcpy(&(sendbufpx[0][0]), &(my_E_prev[1][1]), sizeof(double)*npx); // north
	memcpy(&(sendbufpx[1][0]), &(my_E_prev[npy][1]), sizeof(double)*npx); // south
	for (i = 1; i <= npy; i++){
	  sendbufpy[0][i-1] = my_E_prev[i][1]; // west, prev col
	  sendbufpy[1][i-1] = my_E_prev[i][npx]; // east, next col
	}
	MPI_Irecv(recvbufpx[0], npx, MPI_DOUBLE, prev_row, tag2, row_comm, &reqs[0]); // recv north
	MPI_Irecv(recvbufpx[1], npx, MPI_DOUBLE, next_row, tag1, row_comm, &reqs[1]); // recv south
	MPI_Irecv(recvbufpy[0], npy, MPI_DOUBLE, prev_col, tag4, col_comm, &reqs[2]); // recv west
	MPI_Irecv(recvbufpy[1], npy, MPI_DOUBLE, next_col, tag3, col_comm, &reqs[3]); // recv east

	MPI_Isend(sendbufpx[0], npx, MPI_DOUBLE, prev_row, tag1, row_comm, &reqs[4]); // send for recvr's S
	MPI_Isend(sendbufpx[1], npx, MPI_DOUBLE, next_row, tag2, row_comm, &reqs[5]); // send for recvr's N
	MPI_Isend(sendbufpy[0], npy, MPI_DOUBLE, prev_col, tag3, col_comm, &reqs[6]); // send for recvr's E
	MPI_Isend(sendbufpy[1], npy, MPI_DOUBLE, next_col, tag4, col_comm, &reqs[7]); // send for recvr's W

	t += dt;
	niter++;

	MPI_Waitall(8, reqs, stat);
      } else if (myrowrank == 0 || myrowrank == py - 1){ // North or south cells
	recvpx_i = 0; recvpy_i = 0; reqs_i = 0; sendpx_i = 0; sendpy_i = 0;

	if (myrowrank == 0) { // North cells
	  // Everyone in north will receive south so we don't have to wait for recv
	  //printf("North cell Before recv Myrank :%d recv_i %d reqs_i %d\n", myrank, recv_i, reqs_i);
	  MPI_Irecv(recvbufpx[recvpx_i], npx, MPI_DOUBLE, next_row, tag1, row_comm, &reqs[reqs_i]);
	  recvpx_i++; reqs_i++;

	  // Everyone will send south
	  memcpy(&(sendbufpx[sendpx_i][0]), &(my_E_prev[npy][1]), sizeof(double)*npx);

	  // Everyone will send for receiver's N so we don't have to wait for send
	  MPI_Isend(sendbufpx[sendpx_i], npx, MPI_DOUBLE, next_row, tag2, row_comm, &reqs[reqs_i]);
	  sendpx_i++; reqs_i++;

	} else if (myrowrank == py - 1) { // South cells

	  // Everyone in south will receive from north so we don't have to wait
	  MPI_Irecv(recvbufpx[recvpx_i], npx, MPI_DOUBLE, prev_row, tag2, row_comm, &reqs[reqs_i]);
	  recvpx_i++; reqs_i++;

	  // Everyone will send north
	  memcpy(&(sendbufpx[sendpx_i][0]), &(my_E_prev[1][1]), sizeof(double)*npx);

	  // Everyone will send for receiver's south so we don't have to wait
	  MPI_Isend(sendbufpx[sendpx_i], npx, MPI_DOUBLE, prev_row, tag1, row_comm, &reqs[reqs_i]);
	  sendpx_i++; reqs_i++;

	}

	// Both north cells and south cells will send data to left and right if not at corners

	if(mycolrank != px - 1){ // If not NE or SE cell then it will send data to right
	  // Everyone who isn't NE will send data to right so we don't have to wait for send
	  // also will receive data from right

	  // Receive data from right for its east corner
	  MPI_Irecv(recvbufpy[recvpy_i], npy, MPI_DOUBLE, next_col, tag3, col_comm, &reqs[reqs_i]);
	  recvpy_i++; reqs_i++;

	  for (i = 1; i <= npy; i++)
	    sendbufpy[sendpy_i][i-1] = my_E_prev[i][npx]; // east, next col

	  // Send data to right from its east corner
	  MPI_Isend(sendbufpy[sendpy_i], npy, MPI_DOUBLE, next_col, tag4, col_comm, &reqs[reqs_i]);
	  sendpy_i++; reqs_i++;
	}

	if(mycolrank != 0){ // If not NW or SW cell then it will send data to left

	  // Receive data from left for its west corner
	  MPI_Irecv(recvbufpy[recvpy_i], npy, MPI_DOUBLE, prev_col, tag4, col_comm, &reqs[reqs_i]);
	  recvpy_i++; reqs_i++;

	  for (i = 1; i <= npy; i++)
	    sendbufpy[sendpy_i][i-1] = my_E_prev[i][1]; // west, prev col

	  // Send data to left from its west corner
	  MPI_Isend(sendbufpy[sendpy_i], npy, MPI_DOUBLE, prev_col, tag3, col_comm, &reqs[reqs_i]);
	  sendpy_i++; reqs_i++;
	}

	if((myrowrank == 0 && mycolrank == 0) || // NW
	   (myrowrank == 0 && mycolrank == px - 1) || // NE
	   (myrowrank == py - 1 && mycolrank == 0) || // SW
	   (myrowrank == py - 1 && mycolrank == px - 1) // SE
	   ){
	  t += dt;
	  niter++;

	  MPI_Waitall(4, reqs, stat);
	}else{
	  t += dt;
	  niter++;

	  MPI_Waitall(6, reqs, stat);
	}
      } else { // East or West cells
	// Everyone will receive south so we don't have to wait for recv
	recvpx_i = 0; recvpy_i = 0; reqs_i = 0; sendpx_i = 0; sendpy_i = 0;

	MPI_Irecv(recvbufpx[recvpx_i], npx, MPI_DOUBLE, next_row, tag1, row_comm, &reqs[reqs_i]);
	recvpx_i++; reqs_i++;

	// Everyone will receive north so we don't have to wait for recv
	MPI_Irecv(recvbufpx[recvpx_i], npx, MPI_DOUBLE, prev_row, tag2, row_comm, &reqs[reqs_i]);
	recvpx_i++; reqs_i++;

	if(mycolrank == 0){ // West cells

	  // Everyone in west will receive from east so we don't have to wait
	  MPI_Irecv(recvbufpy[recvpy_i], npy, MPI_DOUBLE, next_col, tag3, col_comm, &reqs[reqs_i]);
	  recvpy_i++; reqs_i++;

	  // Everyone will send east
	  for (i = 1; i <= npy; i++){
	    sendbufpy[sendpy_i][i-1] = my_E_prev[i][npx]; // east, next col
	  }

	  // Everyone will send east so we don't have to wait
	  MPI_Isend(sendbufpy[sendpy_i], npy, MPI_DOUBLE, next_col, tag4, col_comm, &reqs[reqs_i]);
	  sendpy_i++; reqs_i++;

	} else if (mycolrank == px - 1){ // East cells

	  // Everyone in east will receive from west so we don't have to wait
	  MPI_Irecv(recvbufpy[recvpy_i], npy, MPI_DOUBLE, prev_col, tag4, col_comm, &reqs[reqs_i]);
	  recvpy_i++; reqs_i++;

	  // Everyone will send west
	  for (i = 1; i <= npy; i++){
	    sendbufpy[sendpy_i][i-1] = my_E_prev[i][1]; // west, prev col
	  }
	  // Everyone will send west so we don't have to wait
	  MPI_Isend(sendbufpy[sendpy_i], npy, MPI_DOUBLE, prev_col, tag3, col_comm, &reqs[reqs_i]);
	  sendpy_i++; reqs_i++;
	}

	// Everyone will send south
	memcpy(&(sendbufpx[sendpx_i][0]), &(my_E_prev[npy][1]), sizeof(double)*npx);

	// Everyone will send for receiver's N so we don't have to wait for send
	MPI_Isend(sendbufpx[sendpx_i], npx, MPI_DOUBLE, next_row, tag2, row_comm, &reqs[reqs_i]);
	sendpx_i++; reqs_i++;

	// Everyone will send north
       	memcpy(&(sendbufpx[sendpx_i][0]), &(my_E_prev[1][1]), sizeof(double)*npx);
	// Everyone will send for receiver's south so we don't have to wait
	MPI_Isend(sendbufpx[sendpx_i], npx, MPI_DOUBLE, prev_row, tag1, row_comm, &reqs[reqs_i]);
	sendpx_i++; reqs_i++;

	t += dt;
	niter++;

	MPI_Waitall(6, reqs, stat);
      }

      // UNPACK
      if(myrowrank != 0 && myrowrank != py - 1 && mycolrank != 0 && mycolrank != px - 1){ // ordinary cell

	memcpy(&(my_E_prev[0][1]), &(recvbufpx[0][0]), sizeof(double)*npx); // north
	memcpy(&(my_E_prev[npy+1][1]), &(recvbufpx[1][0]), sizeof(double)*npx); // south

	for(i = 1; i <= npy; i++){
	  my_E_prev[i][0] = recvbufpy[0][i-1]; // west corner
	  my_E_prev[i][npx+1] = recvbufpy[1][i-1]; // east corner
	}

      } else if (myrowrank == 0 || myrowrank == py - 1){ // North or south cells
	recvpx_i = 0; recvpy_i = 0;

	if(myrowrank == 0){ // North cells
	  memcpy(&(my_E_prev[npy+1][1]), &(recvbufpx[recvpx_i][0]), sizeof(double)*npx); // south
	  recvpx_i++;
	}else if (myrowrank == py - 1){ // South cells
	  memcpy(&(my_E_prev[0][1]), &(recvbufpx[recvpx_i][0]), sizeof(double)*npx); // north
	  recvpx_i++;
	}

	// Both north and south cells will receive left and right if not at corners

	if(mycolrank != px - 1){
	  for(i = 1; i <= npy; i++){
	    my_E_prev[i][npx+1] = recvbufpy[recvpy_i][i-1]; // east corner
	  }
	  recvpy_i++;
	}

	if(mycolrank != 0){
	  for(i = 1; i <= npy; i++){
	    my_E_prev[i][0] = recvbufpy[recvpy_i][i-1]; // west corner
	  }
	  recvpy_i++;
	}

      } else {
	recvpx_i = 0; recvpy_i = 0;

	// Both west and east cell will receive from south and north
	memcpy(&(my_E_prev[npy+1][1]), &(recvbufpx[recvpx_i][0]), sizeof(double)*npx);
	recvpx_i++;
	memcpy(&(my_E_prev[0][1]), &(recvbufpx[recvpx_i][0]), sizeof(double)*npx);
	recvpx_i++;

	if(mycolrank == 0){ // West cells
	  for(i = 1; i <= npy; i++){
	    my_E_prev[i][npx+1] = recvbufpy[recvpy_i][i-1]; // east corner
	  }
	  recvpy_i++;
	}else if (mycolrank == px - 1){ // East cells
	  for(i = 1; i <= npy; i++){
	    my_E_prev[i][0] = recvbufpy[recvpy_i][i-1]; // west corner
	  }
	  recvpy_i++;
	}
      }

      simulate2D(my_E, my_E_prev, my_R, alpha, npx, npy, kk, dt, a, epsilon, M1, M2, b, num_threads, myrank, P, m);
      //swap current E with previous E
      double **tmp = my_E; my_E = my_E_prev; my_E_prev = tmp;

      /*if (plot_freq){ //edit
	for(i = 1; i <= np; i++)
	  for(j = 1; j <= np ;j++)
	    gather_arr[i-1][j-1] = my_E[i][j];

	MPI_Gather(&(gather_arr[0][0]), m*np, MPI_DOUBLE, &(E[0][myrank*np]), m*np, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if(myrank == 0){
	  int k = (int)(t/plot_freq);
	  if ((t - k * plot_freq) < dt){
	    splot(E,t,niter,m,n);
	  }
	}
      }*/
    }// end of while loop

    if(!no_comm){
      for(i = 1; i<=npy; i++) // We filled gather_arr in this loop with newly calculated values of my_E_prev.
	memcpy(&(gather_arr[i-1][0]), &(my_E_prev[i][1]), sizeof(double)*npx);

         MPI_Gather(&(gather_arr[0][0]), npx*npy, MPI_DOUBLE, &(E[myrowrank*npy][mycolrank*npx]), npx*npy, MPI_DOUBLE, 0, MPI_COMM_WORLD); // We used gather to fill the E array with calculated new values of my_E_prev (To use in stats)
      }

  }

  if(myrank == 0){
    double time_elapsed = getTime() - t0;

    double Gflops = (double)(niter * (1E-9 * n * n ) * 28.0) / time_elapsed ;
    double BW = (double)(niter * 1E-9 * (n * n * sizeof(double) * 4.0  ))/time_elapsed;

    cout << "Number of Iterations        : " << niter << endl;
    cout << "Elapsed Time (sec)          : " << time_elapsed << endl;
    cout << "Sustained Gflops Rate       : " << Gflops << endl;
    cout << "Sustained Bandwidth (GB/sec): " << BW << endl << endl;

    if(!(px == 1 && py == 1)){ // If we don't use 1 process
      for(i = 1; i<=n; i++) // We fill the E_prev with its new values which we gathered.
	for(j = 1; j<=m ;j++)
	  E_prev[i][j] = E[i-1][j-1];
    }

    double mx;
    double l2norm = stats(E_prev,m,n,&mx);
    cout << "Max: " << mx <<  " L2norm: "<< l2norm << endl;

    /*if (plot_freq){
      cout << "\n\nEnter any input to close the program and the plot..." << endl;
      getchar();
    }*/
  }

  // We freed all the allocated memory we used in this program.

  if(px == 1 && py == 1){ // single process
    free(E);
    free(E_prev);
    free(R);
  } else if(px == 1 || py == 1){ // 1D Geometry
    free(E);
    free(E_prev);
    free(R);
    free(my_E);
    free(my_E_prev);
    free(my_R);

    free(sendbuf);
    free(recvbuf);
  } else { // 2D Geometry
    free(E);
    free(E_prev);
    free(R);
    free(my_E);
    free(my_E_prev);
    free(my_R);

    free(sendbufpx);
    free(sendbufpy);
    free(recvbufpx);
    free(recvbufpy);
  }

  MPI_Finalize();

  return 0;
}

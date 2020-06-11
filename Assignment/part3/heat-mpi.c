/*
 * Iterative solver for heat distribution
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "heat.h"

void usage( char *s )
{
    fprintf(stderr, 
	    "Usage: %s <input file> [result file]\n\n", s);
}


copysubmatrix(double * destiny,int destiny_r, int destiny_c , double * source, int source_r, int from_r, int from_c)
{

	for (int i = 0; i < destiny_r; i++) {
		for (int j = 0; j < destiny_c; j++) {
			destiny[i*destiny_r + j] = source[(from_c + i) * source_r + from_r + j];
		}
	}
}

int main( int argc, char *argv[] )
{
    unsigned iter;
    FILE *infile, *resfile;
    char *resfilename;
    int myid, numprocs;
	
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	double* uu[numprocs];
	
    if (myid == 0) {
		printf("I am the master (%d) and going to distribute work to %d additional workers ...\n", myid, numprocs-1);

		// algorithmic parameters
		algoparam_t param;
		int np;

		double runtime, flop;
		double residual=0.0;

		// check arguments
		if( argc < 2 )
		{
		usage( argv[0] );
		return 1;
		}

		// check input file
		if( !(infile=fopen(argv[1], "r"))  ) 
		{
		fprintf(stderr, 
			"\nError: Cannot open \"%s\" for reading.\n\n", argv[1]);
		  
		usage(argv[0]);
		return 1;
		}

		// check result file
		resfilename= (argc>=3) ? argv[2]:"heat.ppm";

		if( !(resfile=fopen(resfilename, "w")) )
		{
		fprintf(stderr, 
			"\nError: Cannot open \"%s\" for writing.\n\n", 
			resfilename);
		usage(argv[0]);
		return 1;
		}

		// check input
		if( !read_input(infile, &param) )
		{
		fprintf(stderr, "\nError: Error parsing input file.\n\n");
		usage(argv[0]);
		return 1;
		}
		print_params(&param);

		// set the visualization resolution
		
		param.u     = 0;
		param.uhelp = 0;
		param.uvis  = 0;
		param.visres = param.resolution;
	   
		if( !initialize(&param) )
		{
			fprintf(stderr, "Error in Solver initialization.\n\n");
			usage(argv[0]);
				return 1;
		}

		// full size (param.resolution are only the inner points)
		np = param.resolution + 2;
		
		// starting time
		runtime = wtime();

		int mp = (np-2)/numprocs + 2;
		// send to workers the necessary data to perform computation
		for (int i=1; i<numprocs; i++) {
			MPI_Send(&param.maxiter, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(&param.resolution, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(&param.algorithm, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			if(i == numprocs-1){
				printf("Sending rows %d to %d to worker %d\n", i * (mp-2), i * (mp-2) + (mp + (np-2)%numprocs), i);
				MPI_Send(&param.u[i * (mp-2) * np], (mp + (np-2)%numprocs)*(np), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
				MPI_Send(&param.uhelp[i * (mp-2) * np], (mp)*(np), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);						
			}
			else{
				printf("Sending rows %d to %d to worker %d\n", i * (mp-2), i * (mp-2) + mp, i);
				MPI_Send(&param.u[i * (mp-2) * np], (mp)*(np), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
				MPI_Send(&param.uhelp[i * (mp-2) * np], (mp)*(np), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
			}
		}
		
		if(param.algorithm == 2)
		{
			for (int b=0; b<numprocs; b++){
				uu[b] = calloc( sizeof(double),(mp)*(mp) );
				copysubmatrix(uu[b], mp, mp, param.u, np, b*(mp-2),0);
			}
		}
		

		iter = 0;
		while(1) {
		switch( param.algorithm ) {
			case 0: // JACOBI
					residual = relax_jacobi(param.u, param.uhelp, mp, np);
				// Copy uhelp into u
				for (int i=0; i<mp; i++)
						for (int j=0; j<np; j++)
						param.u[ i*np+j ] = param.uhelp[ i*np+j ];
					
				MPI_Sendrecv(&param.u[np*(mp-2)], np, MPI_DOUBLE, 1, 0,
					&param.u[np*(mp-1)], np, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &status);
					
				break;
			case 1: // RED-BLACK
				residual = relax_redblack(param.u, np, np);
				break;
			case 2: // GAUSS
				residual=0;
				for (int b=0; b<numprocs; b++){
					residual += relax_gauss(uu[b], mp, mp);
					for(int i=0;i<mp;i++){
						if(b!=numprocs-1) uu[b+1][i*mp] = uu[b][mp*i+mp-2];
						if(b!=0) uu[b-1][i*mp+mp-1] = uu[b][i*mp+1];
					}
					MPI_Send(&uu[b][mp*(mp-2)],mp, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
				}
				for (int b=0; b<numprocs; b++){
					MPI_Recv(&uu[b][mp*(mp-1)], mp, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &status);
				}
				break;
			}

			iter++;

			MPI_Allreduce(MPI_IN_PLACE, &residual, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			
			// solution good enough ?
			if (residual < 0.00005) break;

			// max. iteration reached ? (no limit with maxiter=0)
			if (param.maxiter>0 && iter>=param.maxiter) break;
		}

		
		if(param.algorithm == 2)
			for(int i = 1;i<mp-1;i++)
				for(int j = 1;j<np-1;j++)
						param.u[i*np + j] = uu[(j-1)/(mp-2)][i*mp + (((j-1) % (mp-2))+1)];
		
		for(int i=1; i<numprocs; i++) {
  			MPI_Recv(&param.u[(mp-1 + (i-1) * (mp-2))*np], (mp-2)*np, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
		}
		
		// Flop count after iter iterations
		flop = iter * 11.0 * param.resolution * param.resolution;
		// stopping time
		runtime = wtime() - runtime;

		fprintf(stdout, "Time: %04.3f ", runtime);
		fprintf(stdout, "(%3.3f GFlop => %6.2f MFlop/s)\n", 
			flop/1000000000.0,
			flop/runtime/1000000);
		fprintf(stdout, "Convergence to residual=%f: %d iterations\n", residual, iter);

		// for plot...
		coarsen( param.u, np, np,
			 param.uvis, param.visres+2, param.visres+2 );
	  
		write_image( resfile, param.uvis,  
			 param.visres+2, 
			 param.visres+2 );

		finalize( &param );

		fprintf(stdout, "Process %d finished computing with residual value = %f\n", myid, residual);

		MPI_Finalize();

		return 0;

	} else {

		printf("I am worker %d and ready to receive work to do ...\n", myid);

		// receive information from master to perform computation locally

		int columns, rows, np;
		int iter, maxiter;
		int algorithm;
		double residual;

		MPI_Recv(&maxiter, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&columns, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&algorithm, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

		rows = columns;
		np = columns + 2;
		int mp = (np-2)/numprocs + 2;

		int last_process = myid == numprocs-1;
		
		
		if (last_process)
		{
			mp = mp + (np-2)%numprocs;
		}

		
		
		
		// allocate memory for worker
		double * u = calloc( sizeof(double),(mp)*(np) );
		double * uhelp = calloc( sizeof(double),(mp)*(np) );
		if( (!u) || (!uhelp) )
		{
			fprintf(stderr, "Error: Cannot allocate memory\n");
			return 0;
		}
		
		// fill initial values for matrix with values received from master
		MPI_Recv(u, (mp)*(np), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(uhelp, (mp)*(np), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);

		
		
		if(algorithm == 2)
		{
			for (int b=0; b<numprocs; b++){
				uu[b] = calloc( sizeof(double),(mp)*(mp) );
				copysubmatrix(uu[b], mp, mp, u, np, b*(mp-2),0);
			}
		}
		
		
		iter = 0;
		while(1) {
		switch( algorithm ) {
	    case 0: // JACOBI
			residual = relax_jacobi(u, uhelp, mp, np);
		    // Copy uhelp into u
		    for (int i=0; i<mp; i++)
    		        for (int j=0; j<np; j++)
						u[ i*np+j ] = uhelp[ i*np+j ];
			
			if (!last_process) MPI_Send(&u[np*(mp-2)], np, MPI_DOUBLE, myid+1, 0, MPI_COMM_WORLD);
			MPI_Recv(u, np, MPI_DOUBLE, myid-1, 0, MPI_COMM_WORLD, &status);
			MPI_Send(&u[np], np, MPI_DOUBLE, myid-1, 0, MPI_COMM_WORLD);	
			if (!last_process) MPI_Recv(&u[np*(mp-1)], np, MPI_DOUBLE, myid+1, 0, MPI_COMM_WORLD, &status);
		    break;
	    case 1: // RED-BLACK
		    residual = relax_redblack(u, np, np);
		    break;
	    case 2: // GAUSS
			residual=0;
			for (int b=0; b<numprocs; b++){
				MPI_Recv(uu[b],mp, MPI_DOUBLE, myid-1, 0, MPI_COMM_WORLD, &status);
				residual += relax_gauss(uu[b], mp, mp);
				for(int i=0;i<mp;i++){
					if(b!=numprocs-1) uu[b+1][i*mp] = uu[b][mp*i+mp-2];
					if(b!=0) uu[b-1][i*mp+mp-1] = uu[b][i*mp+1];
				}
				if(!last_process) MPI_Send(&uu[b][mp*(mp-2)],mp, MPI_DOUBLE, myid+1, 0, MPI_COMM_WORLD);
			}
			for (int b=0; b<numprocs; b++){
				MPI_Send(&uu[b][mp],mp, MPI_DOUBLE, myid-1, 0, MPI_COMM_WORLD);
				if(!last_process) MPI_Recv(&uu[b][mp*(mp-1)], mp, MPI_DOUBLE, myid+1, 0, MPI_COMM_WORLD, &status);
			}
		    break;
	    }

        iter++;

		MPI_Allreduce(MPI_IN_PLACE, &residual, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		
        // solution good enough ?
        if (residual < 0.00005) break;

        // max. iteration reached ? (no limit with maxiter=0)
        if (maxiter>0 && iter>=maxiter) break;
		}
		
		if(algorithm == 2)
			for(int i = 1;i<mp-1;i++)
				for(int j = 1;j<np-1;j++)
						u[i*np + j] = uu[(j-1)/(mp-2)][i*mp + (((j-1) % (mp-2))+1)];
		
		MPI_Send(&u[np], (mp-2)*np, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

		if( u ) free(u); if( uhelp ) free(uhelp);

		fprintf(stdout, "Process %d finished computing %d iterations with residual value = %f\n", myid, iter, residual);

		MPI_Finalize();
		exit(0);
	}
}

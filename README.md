# CPDS-M2

Code used to copy files. It uses Putty scp command line utility.
```
> pscp -r nct01036@mt1.bsc.es:/home/nct01/nct01036/Assignment .
```
To copy files back to the machine, run from inside the CPDS-M2 root folder:

```
> pscp -r Assignment nct01036@mt1.bsc.es:/home/nct01/nct01036/
```

To run part2:

```
> export OMP_NUM_THREADS=4
> ./heatomp test.dat
```

To run part3:

```
> mpiexec -n 4 heatmpi test.dat
```

To run part4:

```
> export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cuda/4.1/lib64
> ./heatCUDA test.dat -t 16
```
# CPDS-M2

Code used to copy files. It uses Putty scp command line utility.
```
> pscp -r nct01036@mt1.bsc.es:/home/nct01/nct01036/Assignment .
```
To copy files back to the machine, run from inside the CPDS-M2 root folder:

```
> pscp -r Assignment nct01036@mt1.bsc.es:/home/nct01/nct01036/
```

To run part3:

```
> mpiexec -n 4 heatmpi test.dat
```

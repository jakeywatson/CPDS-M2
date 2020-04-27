CC     =  gcc
CC     =  icc

OPT3 	= -O3
CFLAGS  = -qopenmp -Wall -std=c99
LFLAGS  = -lm

ALL	= heatomp
all: $(ALL)

misc.o: misc.c
	$(CC) -c $(CFLAGS) $(OPT3) $< -o $@

heatomp: heat-omp.c solver-omp.c misc.o
	$(CC) $(CFLAGS) $(OPT3) -qopenmp $+ $(LFLAGS) -o $@

clean:
	rm -fr $(ALL) heat.ppm *.o *.out *.err *.prv *.pcf *.row *.sym TRACE.mpits set-0 submit-omp.sh.o* submit-omp.sh.e* submit-omp-i.sh.o* submit-omp-i.sh.e*

include ../../config.mak

OBJ = write_ppm.o read_options.o

all: $(OBJ)

$(OBJ): ../../config.mak

write_ppm.o: write_ppm.f90
	$(FC) -c $(FFLAGS) $<

read_options.o: read_options.f90
	$(FC) -c $(FFLAGS) $<
clean:
	-rm $(OBJ) *.mod

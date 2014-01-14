#ifndef READ_OPTIONS_H
#define READ_OPTIONS_H

void read_options(int, char**, int*, int*, int *,
                  float*, float*, float*, float*);


/* default coordinates for calculation */
#define XMIN (-2.0)
#define XMAX (1.0)
#define YMIN (-1.2)
#define YMAX (1.2)

/* size of entire grid in X dimension (Y dimension is scaled
 * appropriately) */
#define GRIDSIZE_X 768

/* default number of iterations */
#define ITERATIONS 5000

#endif

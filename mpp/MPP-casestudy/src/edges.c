#include <stdio.h>
#include <stdlib.h>
#include "pgmio.h"


int main(int argc, char *argv[])
{
        /** The image filename is provided as the first
         *  parameter to the program.
         */
        if (argc < 4) {
                printf("This program is run as follows: %s image_name dim_x dim_y",
                                argv[0]);
                return -1;
        }
        // Parsing the command line parameters
        char **char_ptr = NULL;
        char *image_name = argv[1];
        size_t dim_x = strtoul(argv[2], char_ptr, 10);
        size_t dim_y = strtoul(argv[3], char_ptr, 10);

        float *image_data = malloc(sizeof(float) * dim_x * dim_y);
        pgmread(image_name, image_data, dim_x, dim_y);



        return 0;
}

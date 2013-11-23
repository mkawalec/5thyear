/**
 *  @file   helpers.h
 *  @Author Michal Kawalec (michal@bazzle.me)
 *  @date   November, 2013
 *  @brief  A set of functions helping with 2D decompositions
 *          using MPI
 */

#ifndef helpers_h
#define helpers_h

#include <mpi.h>

/**
 *  @brief A simple pair containing two elements.
 */
struct pair {
    int first, second;
};

/**
 *  @brief          Reads in buf into a 2D array new.
 *  @param buf      input image in 1D array
 *  @param new      an output 2D array of size (dim_x + 2)x(dim_y + 2)
 *  @param dim_x    X dimension of buf
 *  @param dim_y    Y dimension of buf
 *
 *
 *  Pads the array with a maximum value (255) in the
 *  halo area (the border).
 */
void read_input(float *buf, float **new, size_t dim_x, size_t dim_y);

/**
 *  @brief          Sets a 2D array to 255 everywhere
 *  @param array    a 2D array values of which will be set to 255
 *  @param dim_x    X dimension of array
 *  @param dim_y    Y dimension of array
 */
void initialize_array(float **array, size_t dim_x, size_t dim_y);

/**
 *  @brief          Computes sum of all non-halo elements in an array
 *  @param array    a 2D array containing halos
 *  @param dim_x    an X size of the array inner elements (witout the halos)
 *  @param dim_y    an Y size of the array
 *  @return         a sum of values
 */
unsigned long long compute_sum(float **array, size_t dim_x, size_t dim_y);

/**
 *  @brief          Computes the total parts circumference for 
 *                  a given decomposition.
 *  @param dim_x    X dimension of a grid sliced into parts
 *  @param dim_y    Y dimension of a grid sliced into parts
 *  @param part_x   highest X size of a decomposition part
 *  @param part_y   highest Y size of a decomposition part
 *
 *
 *  By total cincumference we mean the complete amout of data
 *  that has to be sent every iteration step, as the data being
 *  sent resides in halos which size is proportional to the
 *  circumference of each part.
 */
size_t circ(size_t dim_x, size_t dim_y, size_t part_x, size_t part_y);

/**
 *  @brief          Provides coordinates in pixels of an image part designated
 *                  for processing by process with a rank.
 *  @param rank     rank of a process for which coordinates are requested
 *  @param dim_x    X dimension of a grid
 *  @param dim_y    Y dimension of a grid
 *  @param part_x   X size of an optimal decomposition
 *  @param part_y   Y size of an optimal decomposition
 *  @param start_x  X coordinate of a lower-left corner of requested part
 *  @param start_y  Y coordinate of a lower-left corner of requested part
 *  @param end_x    X coordinate of an upper-right corner of requested part
 *  @param end_y    Y coordinate of an upper-right corner of requested part
 *
 *
 *  Note that this returns the position assuming C-like part
 *  numbering (the numbering IS different in MPI!). The
 *  data inside a part needs to be inverted to be correctly
 *  processed with MPI.
 */
void get_pos(int rank, size_t dim_x, size_t dim_y, size_t part_x, size_t part_y,
        size_t *start_x, size_t *start_y, size_t *end_x, size_t *end_y);

/**
 *  @brief          Returns a virtual datatype for part being sent
 *                  to a process with rank.
 *  @brief rank     rank of a target process
 *  @brief dim_x    X dimension of data grid
 *  @brief dim_y    Y dimension of data grid
 *  @brief size_x   X size of an optimal decomposition
 *  @brief size_y   Y size of an optimal decomposition
 *
 *
 *  Rember that the returned datatype is uncommitted, for it to
 *  be used MPI_Type_commit needs to be called.
 */
MPI_Datatype create_dtype(int rank, size_t dim_x, size_t dim_y, 
        size_t size_x, size_t size_y);

/**
 *  @brief          Performs a virtual inversion on a buffer
 *  @param buffer   pointer to a pointer to a 1D float array
 *  @param height   height of a buffer row
 *  @param size     number of elements in an array buffer points to
 *
 *
 *  The inversion is performed in such a way that
 *  after the function finishes buffer is a pointer to 
 *  a pointer to an inverted array
 */
void invert(float **buffer, int height, int size);

/** 
 *  @brief              Performs a 2D scatter of input to receive_buffer over 
 *                      a specified communicator.
 *  @param input        input data in a list of floats
 *  @param dim_x        X size of the input data
 *  @param dim_y        Y size of the input data
 *  @param communicator communicator used for data transfer
 *  @param receive_buf  a pointer to a pointer to a receive buffer
 *  @param receive_x    X dimension of the receive buffer
 *  @param receive_y    Y dimension of the receive buffer
 *
 *
 *  We were unsure if there was any benefit of making the function
 *  more general and make it accept any filetype. It was decided that
 *  simplicity of this particular code is more important than 
 *  the possible gains for other using this code.
 *  If this was a code written with the intention of being reused,
 *  the following function would surely be written in a general form
 */
void my_scatter(float *input, size_t dim_x, size_t dim_y, MPI_Comm communicator,
                float **receive_buf, size_t *receive_x, size_t *receive_y);

/**
 *  @brief              Performs a 2D gather for input to receive_buf over
 *                      a specified communicator.
 *  @param input        input array
 *  @param input_x      the size of an input array in X direction
 *  @param input_y      the size of an input array in Y direction
 *  @param communicator communicator used for data transfer
 *  @param receive_buf  the buffer to which the the inputs are gathered
 *  @param receive_x    the X size of the array to which data is gathered
 *  @param receive_y    the Y size of the array to which data is gathered
 *
 *  The function assumes that it is used in conjuction with
 *  my_scatter, correctness cannot be guaranteed in other
 *  cases.
 */
void my_gather(float *input, size_t input_x, size_t input_y, MPI_Comm communicator,
               float *receive_buf, size_t receive_x, size_t receive_y);

/**
 *  @brief              Returns an optimal length of a decomposition part
 *                      in X and Y directions
 *  @param dim_x        X size of the decomposed array
 *  @param dim_y        Y size of the decomposed array
 *  @param comm_size    the number of nodes taking part in the decomposition
 *  @return             a pair of numbers describing a length of a part
 *                      in both X (first) and Y (second) directions
 */
struct pair get_decomposition_length(size_t dim_x, size_t dim_y, int comm_size);

/**
 *  @brief              Returns a number of decomposition parts in
 *                      X and Y directions
 *  @param dim_x        X size of the decomposed array
 *  @param dim_y        Y size of the decomposed array
 *  @param comm_size    number of nodes taking part in the decomposition
 *  @return             a pair of numbers, the first of which describes the
 *                      number of nodes in X and the second in Y directions.
 */
struct pair get_decomposition_size(size_t dim_x, size_t dim_y, int comm_size);

/**
 *  @brief          Computes max difference between 
 *                  pixel values on two images.
 *  @param old      one image in a 2D array (with halos)
 *  @param new      another image in a 2D array (with halos)
 *  @param dim_x    X direction size of both images (excluding halos)
 *  @param dim_y    Y direction size of both images (excluding halos)
 *  @return         maximum difference between pixels at 
 *                  the same coordinates on both images
 */
float compute_max_change(float **old, float **new, size_t dim_x, size_t dim_y);

/**
 *  @brief  Returns a current time in seconds,
 *          up to a reasonable accuracy
 */
double get_time();

#endif

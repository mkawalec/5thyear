/**
 *  @file   comm.h
 *  @Author Michal Kawalec (michal@bazzle.me)
 *  @date   November, 2013
 *  @brief  Functions responsible for communication
 *          when doing a 2D decomposition with MPI
 */

#ifndef comm_h
#define comm_h

#include <mpi.h>
#include <stddef.h>

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

#endif

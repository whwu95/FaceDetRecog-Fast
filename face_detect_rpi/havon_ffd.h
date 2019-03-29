#ifndef HAVON_XFFD_H
#define HAVON_XFFD_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif 

struct square_rect {
    uint32_t cx, cy, size;
    float score;
};

struct havon_xffd;
struct havon_xffd* havon_xffd_create(uint32_t min_size, uint32_t max_size);
void havon_xffd_destroy(struct havon_xffd *xffd);    

/**
 *  \brief detect faces in gray scale image
 *
 *  \param xffd valid detector instance
 *  \param graydata gray image data 
 *  \param width width of gray scale image 
 *  \param height height of the gray scale image 
 *  \param step stride of widthstep of gray scale image 
 *  \param buff buffer used to store detection results 
 *  \param buff_size capacity of buff 
 *  \param num_saved number of faces put into the buff
 *  \return return negative values for errors, 0 for success 
 */
int32_t havon_xffd_detect(struct havon_xffd *xffd, const uint8_t* graydata,
                          uint32_t width, uint32_t height, uint32_t step,
                          struct square_rect buff[], uint32_t buff_size, uint32_t *num_saved);


#ifdef __cplusplus
}
#endif 

#endif /* HAVON_XFFD_H */

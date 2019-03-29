#include "havon_ffd_cascade.h"
#include "havon_ffd.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus 
extern "C" {
#endif 

#define MAX_PYRAMID_NUM 6
    
struct havon_xffd {
    uint32_t min_size, max_size;
    struct cascade_trees *cascade;
    struct model *model;
    const uint8_t *gray;
    uint32_t width, height, widthstep;
};
    
struct havon_xffd*
havon_xffd_create(uint32_t min_size, uint32_t max_size) {
    struct havon_xffd *xffd = (struct havon_xffd *)malloc(sizeof(*xffd));
    if (xffd) {
        xffd->min_size = min_size;
        xffd->max_size = max_size;
        xffd->cascade = &__cascade;
        // image 
        xffd->gray = NULL;
        xffd->width = 0;
        xffd->height = 0;
        xffd->widthstep = 0;
    }
    return xffd;
}

void havon_xffd_destroy(struct havon_xffd *xffd) {
    if (xffd) {
        free(xffd);
    }
}

static uint32_t
cascading(const struct havon_xffd *xffd, uint32_t r, uint32_t c, uint32_t size, float *score) {    
    const struct cascade_trees *cascade = xffd->cascade;
    const uint32_t tree_depth = cascade->tree_depth;
    const uint32_t wstep = xffd->widthstep;
    // const uint32_t width = xffd->width, height = xffd->height;
    const uint8_t *gray = xffd->gray;
    
    float ss = 0.f;    
    uint32_t i, j;

    
    r = r*256;
    c = c*256;    
    
    /* if( (r+128*size)/256>=height || (r-128*size)/256<0 || (c+128*size)/256>=width || (c-128*size)/256<0 ) */
	/* 	return 0; */

    for (i = 0; i < cascade->num_trees; ++i) {
        const float thresh = cascade->trees[i].thresh;
        const int8_t *tcodes = &(cascade->trees[i].tcodes[0]);
        const float *lut = &(cascade->trees[i].lut[0]);
        
        uint32_t idx = 1; 
        for (j = 0; j < tree_depth; ++j) {            
            idx = 2*idx + (gray[(r+tcodes[4*idx+0]*size)/256*wstep+(c+tcodes[4*idx+1]*size)/256]
                           <=
                           gray[(r+tcodes[4*idx+2]*size)/256*wstep+(c+tcodes[4*idx+3]*size)/256]);
        }

        ss = ss + lut[idx - (1 << tree_depth)];                
        if (ss < thresh) {
            return 0; // nothing detected 
        }
    }
    
    *score = ss; // may need subtract latest thresh 
    return 1; // detected one 
}

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) > (b) ? (b) : (a))


static float
get_overlap(float r1, float c1, float s1, float r2, float c2, float s2) {
    const float overr = MAX(0, MIN(r1+s1/2, r2+s2/2) - MAX(r1-s1/2, r2-s2/2));
    const float overc = MAX(0, MIN(c1+s1/2, c2+s2/2) - MAX(c1-s1/2, c2-s2/2));
    return overr*overc/(s1*s1+s2*s2-overr*overc);
}


static void
ccdfs(int a[], int i, const struct square_rect *rects, int n) {
    int j;
    for(j=0; j<n; ++j) {
        if(a[j]==0 &&
           get_overlap(rects[i].cx, rects[i].cy, rects[i].size, rects[j].cx, rects[j].cy, rects[j].size) > 0.3f) {
            a[j] = a[i];
            ccdfs(a, j, rects, n);
        }
    }
}

static int
find_connected_components(int a[], const struct square_rect *rects, int n) {
    int i, cc;
    if(!n) return 0;
    for(i=0; i<n; ++i)
        a[i] = 0;
    cc = 1;
    for(i=0; i<n; ++i) {
        if(a[i] == 0) {
            a[i] = cc;
            ccdfs(a, i, rects, n);
            ++cc;
        }
    }
    return cc - 1; // number of connected components
}

static uint32_t
cluster_detections(struct square_rect *rects, uint32_t n) {
    int idx, ncc, cc;
    int a[4096];

    ncc = find_connected_components(a, rects, n);
    
    if(!ncc) return 0;

    idx = 0;

    for(cc=1; cc<=ncc; ++cc) {
        uint32_t i, k;
        float sumqs=0.0f, sumrs=0.0f, sumcs=0.0f, sumss=0.0f;
        k = 0;
        for(i=0; i<n; ++i)
            if(a[i] == cc) {
                sumcs += rects[i].cx;
                sumrs += rects[i].cy;
                sumss += rects[i].size;
                sumqs += rects[i].score;
                ++k;
            }

        rects[idx].cx = sumcs/k;
        rects[idx].cy = sumrs/k;
        rects[idx].size = sumss/k;;
        rects[idx].score = sumqs; // accumulated confidence measure
        ++idx;
    }    
    return idx;
}


int32_t
havon_xffd_detect(struct havon_xffd *xffd, const uint8_t* graydata,
                  uint32_t width, uint32_t height, uint32_t widthstep,
                  struct square_rect buff[], uint32_t buff_size, uint32_t *num_saved) {

    int32_t ret = 0;
    const uint32_t min_size = xffd->min_size, max_size = xffd->max_size;
    uint32_t detected_cnt = 0;
    uint32_t s, r, c;
    
    xffd->gray = graydata;
    xffd->width = width;
    xffd->height = height;
    xffd->widthstep = widthstep;

    
    for (s = min_size; s <= max_size;) {        
        const uint32_t face_size_step = (uint32_t)(s * 0.1f + 0.5);
        for (r = s/2 + 1; r <= height-s/2-1; r += face_size_step) {
            for (c = s/2 + 1; c <= width-s/2-1; c += face_size_step) {
                float score = 0;
                if (cascading(xffd, r, c, s, &score)) {
                    if (detected_cnt < buff_size) {
                        buff[detected_cnt].cx = c;
                        buff[detected_cnt].cy = r;
                        buff[detected_cnt].size = s;
                        buff[detected_cnt].score = score;
                        ++detected_cnt;
                    }
                    else {
                        ret = 1;
                        goto leave;
                    }
                }
            }
        }
        s += face_size_step;
    }

 leave:
    *num_saved = cluster_detections(buff, detected_cnt);
    // no clustering 
    // *num_saved = detected_cnt;
    return ret;
}
    
#ifdef __cplusplus 
}
#endif 


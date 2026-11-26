// Minimal SIFT example: sift_min <img1> [img2]
#include <stdio.h>
#include <stdlib.h>

#include "improc.h"

#define MAX_KPS 20000

struct Features {
    struct SiftKeyPoint *kps;
    uint8_t (*desc)[SIFT_DESC_BYTES];
    int32_t n;
};

static int detect(const char *path, struct Features *f) {
    struct Mat *img = alloc_mat_from_file(path);
    if (img == NULL) {
        return -1;
    }
    struct Mat *gray = img;
    const int   cn   = (int)mat_channel_num(img);
    if (cn == 3 || cn == 4) {
        gray = alloc_mat(img->rows, img->cols, CV_8UC1);
        cvt_color(gray, img, cn == 3 ? COLOR_RGB2GRAY : COLOR_RGBA2GRAY);
    } else if (cn != 1) {
        fprintf(stderr, "%s: %d channels not supported\n", path, cn);
        free_mat(&img);
        return -1;
    }
    f->kps  = malloc(MAX_KPS * sizeof *f->kps);
    f->desc = malloc(MAX_KPS * sizeof *f->desc);
    if (f->kps == NULL || f->desc == NULL) {
        return -1;
    }
    f->n = sift_detect_compute(f->kps, f->desc, MAX_KPS, gray, NULL);
    printf("%s: %dx%d, %ld keypoints%s\n", path, gray->cols, gray->rows,
           (long)f->n, f->n == MAX_KPS ? " (capacity reached)" : "");
    for (int32_t i = 0; i < f->n && i < 5; i++) {
        const struct SiftKeyPoint *k = &f->kps[i];
        printf("  x %.2f y %.2f size %.2f angle %.1f response %.4f\n",
               (double)k->x, (double)k->y, (double)k->size, (double)k->angle,
               (double)k->response);
    }
    if (gray != img) {
        free_mat(&gray);
    }
    free_mat(&img);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: sift_min <img1> [img2]\n");
        return 2;
    }
    struct Features a = {0}, b = {0};
    if (detect(argv[1], &a) != 0 || (argc == 3 && detect(argv[2], &b) != 0)) {
        return 1;
    }
    if (argc == 3) {
        struct SiftMatch *m = malloc((a.n > 0 ? (size_t)a.n : 1) * sizeof *m);
        if (m == NULL) {
            return 1;
        }
        const int32_t nm = sift_match(m, a.desc, a.n, b.desc, b.n, 0.8f);
        printf("matches (ratio 0.8): %ld\n", (long)nm);
        free(m);
    }
    free(a.kps);
    free(a.desc);
    free(b.kps);
    free(b.desc);
    return 0;
}

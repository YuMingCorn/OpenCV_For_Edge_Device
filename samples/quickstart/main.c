// Grayscale, 5x5 Gaussian blur and Otsu threshold on one image.
//   quickstart [input.png|input.jpg] [output.png]
// Without an input it uses the built-in 240x320 test pattern.
#include <stdio.h>

#include "improc.h"

int main(int argc, char **argv) {
    const char *out_path = argc > 2 ? argv[2] : "binary.png";
    const char *in_path  = argc > 1 ? argv[1] : NULL;
    struct Mat *src =
        in_path ? alloc_mat_from_file(in_path) : alloc_mat_synthetic(240, 320);
    if (src == NULL) {
        fprintf(stderr, "cannot read %s\n", in_path ? in_path : "test pattern");
        return 1;
    }

    struct Mat *gray = alloc_mat(src->rows, src->cols, CV_8UC1);
    struct Mat *blur = alloc_mat(src->rows, src->cols, CV_8UC1);
    struct Mat *bin  = alloc_mat(src->rows, src->cols, CV_8UC1);

    cvt_color(gray, src, COLOR_RGB2GRAY);
    gaussian_blur(blur, gray, 5);
    uchar t = otsu_threshold(blur);
    threshold(bin, blur, t, 255, THRESH_BINARY);

    int status = mat_write(out_path, bin) == 0 ? 0 : 1;
    if (status == 0) {
        printf("%dx%d, Otsu threshold %u, wrote %s\n", src->cols, src->rows,
               (unsigned)t, out_path);
    } else {
        fprintf(stderr, "cannot write %s\n", out_path);
    }

    free_mat(&bin);
    free_mat(&blur);
    free_mat(&gray);
    free_mat(&src);
    return status;
}

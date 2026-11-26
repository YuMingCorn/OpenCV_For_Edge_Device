// Runs one improc op on a raw image file, for tools/sweep_opencv.py.
//
//   improc_probe <in.raw> <rows> <cols> <in_type> <out.raw> <op> [args]
//
// in_type: u8c1, u8c2, u8c3, u8c4, s16c1, s16c3, f32c1. The raw files hold the
// rows back to back with no padding, in native byte order. Ops:
//
//   cvt <CODE>                  CODE is a ColorConversionCodes name without
//                               the COLOR_ prefix, e.g. RGB2GRAY
//   scale <alpha_q8> <beta>     convert_scale (u8 in)
//   scale_abs <alpha_q8> <beta> convert_scale_abs (s16 in)
//   resize <dst_cols> <dst_rows> <nearest|linear|area>
//   equalize                    equalize_hist (u8c1 in)
//   to_u8                       convert_to u8 with alpha 1, beta 0 (f32c1 in)
//
// Exit status 0 on success, 2 on a usage or IO error.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "improc.h"

struct CodeName {
    const char *name;
    int         code;
    int         dcn;
};

static const struct CodeName codes[] = {
    {"RGB2GRAY", COLOR_RGB2GRAY, 1},
    {"BGR2GRAY", COLOR_BGR2GRAY, 1},
    {"RGBA2GRAY", COLOR_RGBA2GRAY, 1},
    {"BGRA2GRAY", COLOR_BGRA2GRAY, 1},
    {"RGB2HSV", COLOR_RGB2HSV, 3},
    {"HSV2RGB", COLOR_HSV2RGB, 3},
    {"HSV2BGR", COLOR_HSV2BGR, 3},
    {"RGB2YUV", COLOR_RGB2YUV, 3},
    {"BGR2YUV", COLOR_BGR2YUV, 3},
    {"YUV2RGB", COLOR_YUV2RGB, 3},
    {"YUV2BGR", COLOR_YUV2BGR, 3},
    {"BGR5652GRAY", COLOR_BGR5652GRAY, 1},
    {"BGR565BE2GRAY", COLOR_BGR565BE2GRAY, 1},
};

static int usage(void) {
    fprintf(stderr, "usage: improc_probe <in.raw> <rows> <cols> <in_type> "
                    "<out.raw> <op> [args]\n");
    return 2;
}

static int parse_type(const char *s) {
    static const struct {
        const char *name;
        int         type;
    } types[] = {{"u8c1", CV_8UC1},  {"u8c2", CV_8UC2},   {"u8c3", CV_8UC3},
                 {"u8c4", CV_8UC4},  {"s16c1", CV_16SC1}, {"s16c3", CV_16SC3},
#if IMPROC_HAS_FLOAT
                 {"f32c1", CV_32FC1}
#endif
    };
    for (size_t i = 0; i < sizeof types / sizeof types[0]; i++) {
        if (strcmp(s, types[i].name) == 0) {
            return types[i].type;
        }
    }
    return -1;
}

static struct Mat *read_raw(const char *path, int rows, int cols, int type) {
    struct Mat *m = mat_create(rows, cols, type);
    if (m == NULL) {
        return NULL;
    }
    FILE  *f    = fopen(path, "rb");
    size_t want = MAT_DATA_SIZE(rows, cols, type);
    if (f == NULL || fread(m->data, 1, want, f) != want) {
        if (f != NULL) {
            fclose(f);
        }
        free_mat(&m);
        return NULL;
    }
    fclose(f);
    return m;
}

static int write_raw(const char *path, const struct Mat *m) {
    FILE  *f    = fopen(path, "wb");
    size_t size = MAT_DATA_SIZE(m->rows, m->cols, (int)mat_type(m));
    if (f == NULL) {
        return -1;
    }
    int ok = fwrite(m->data, 1, size, f) == size;
    return fclose(f) == 0 && ok ? 0 : -1;
}

static struct Mat *run_op(const struct Mat *src, int argc, char **argv) {
    const char *op = argv[0];
    if (strcmp(op, "cvt") == 0 && argc == 2) {
        for (size_t i = 0; i < sizeof codes / sizeof codes[0]; i++) {
            if (strcmp(argv[1], codes[i].name) == 0) {
                struct Mat *dst = alloc_mat(src->rows, src->cols,
                                            CV_MAKETYPE(CV_8U, codes[i].dcn));
                cvt_color(dst, src, codes[i].code);
                return dst;
            }
        }
        return NULL;
    }
    if ((strcmp(op, "scale") == 0 || strcmp(op, "scale_abs") == 0) &&
        argc == 3) {
        int32_t     alpha = (int32_t)strtol(argv[1], NULL, 10);
        int32_t     beta  = (int32_t)strtol(argv[2], NULL, 10);
        int         cn    = (int)mat_channel_num(src);
        struct Mat *dst =
            alloc_mat(src->rows, src->cols, CV_MAKETYPE(CV_8U, cn));
        if (op[5] == '\0') {
            convert_scale(dst, src, alpha, beta);
        } else {
            convert_scale_abs(dst, src, alpha, beta);
        }
        return dst;
    }
    if (strcmp(op, "resize") == 0 && argc == 4) {
        int dw     = (int)strtol(argv[1], NULL, 10);
        int dh     = (int)strtol(argv[2], NULL, 10);
        int interp = strcmp(argv[3], "nearest") == 0  ? INTER_NEAREST
                     : strcmp(argv[3], "linear") == 0 ? INTER_LINEAR
                     : strcmp(argv[3], "area") == 0   ? INTER_AREA
                                                      : -1;
        if (interp < 0) {
            return NULL;
        }
        struct Mat *dst = alloc_mat(dh, dw, (int)mat_type(src));
        resize(dst, src, interp);
        return dst;
    }
    if (strcmp(op, "equalize") == 0 && argc == 1 &&
        mat_type(src) == (size_t)CV_8UC1) {
        struct Mat *dst = alloc_mat(src->rows, src->cols, CV_8UC1);
        equalize_hist(dst, src);
        return dst;
    }
#if IMPROC_HAS_FLOAT
    if (strcmp(op, "to_u8") == 0 && argc == 1 &&
        mat_type(src) == (size_t)CV_32FC1) {
        struct Mat *dst = alloc_mat(src->rows, src->cols, CV_8UC1);
        convert_to(dst, src, 1.0f, 0.0f);
        return dst;
    }
#endif
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 7) {
        return usage();
    }
    int rows = (int)strtol(argv[2], NULL, 10);
    int cols = (int)strtol(argv[3], NULL, 10);
    int type = parse_type(argv[4]);
    if (rows <= 0 || cols <= 0 || type < 0) {
        return usage();
    }
    struct Mat *src = read_raw(argv[1], rows, cols, type);
    if (src == NULL) {
        fprintf(stderr, "cannot read %s\n", argv[1]);
        return 2;
    }
    struct Mat *dst = run_op(src, argc - 6, argv + 6);
    free_mat(&src);
    if (dst == NULL) {
        return usage();
    }
    int rc = write_raw(argv[5], dst);
    free_mat(&dst);
    if (rc != 0) {
        fprintf(stderr, "cannot write %s\n", argv[5]);
        return 2;
    }
    return 0;
}

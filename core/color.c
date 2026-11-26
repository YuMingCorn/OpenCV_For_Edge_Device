#include "color.h"

#include <stdint.h>
#include <string.h>

// HSV2RGB can use the hardware's float32 directly on 64-bit hosts whose float
// arithmetic is IEEE single precision evaluated in float. The result is the
// same as the integer emulation below, which every other target uses (and
// IMPROC_NO_SIMD forces, so the tests also run it on a host).
#if IMPROC_HAS_FLOAT && !defined(IMPROC_NO_SIMD) &&                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__))
#include <float.h>
#if FLT_EVAL_METHOD == 0
#define IMPROC_HSV_FLOAT 1
#endif
#endif
#ifndef IMPROC_HSV_FLOAT
#define IMPROC_HSV_FLOAT 0
#endif

// OpenCV's Q15 gray weights (RGB2GRAY and BGR5652GRAY). The sum reaches
// 32768 * 255 + 16384, so it needs uint32_t where int is 16 bits.
static inline uchar luma(uint32_t r, uint32_t g, uint32_t b) {
    return (uchar)((9798u * r + 19235u * g + 3735u * b + 16384u) >> 15);
}

static void rgb2gray_row(const uchar *restrict s, uchar *restrict d,
                         int width) {
    for (int x = 0; x < width; x++, s += 3) {
        d[x] = luma(s[0], s[1], s[2]);
    }
}

static void bgr2gray_row(const uchar *restrict s, uchar *restrict d,
                         int width) {
    for (int x = 0; x < width; x++, s += 3) {
        d[x] = luma(s[2], s[1], s[0]);
    }
}

static void rgba2gray_row(const uchar *restrict s, uchar *restrict d,
                          int width) {
    for (int x = 0; x < width; x++, s += 4) {
        d[x] = luma(s[0], s[1], s[2]);
    }
}

static void bgra2gray_row(const uchar *restrict s, uchar *restrict d,
                          int width) {
    for (int x = 0; x < width; x++, s += 4) {
        d[x] = luma(s[2], s[1], s[0]);
    }
}

static void gray2rgb_row(const uchar *restrict s, uchar *restrict d,
                         int width) {
    for (int x = 0; x < width; x++, d += 3) {
        d[0] = d[1] = d[2] = s[x];
    }
}

static void gray2rgba_row(const uchar *restrict s, uchar *restrict d,
                          int width) {
    for (int x = 0; x < width; x++, d += 4) {
        d[0] = d[1] = d[2] = s[x];
        d[3]               = 255;
    }
}

static void rgb2bgr_row(const uchar *restrict s, uchar *restrict d, int width) {
    for (int x = 0; x < width; x++, s += 3, d += 3) {
        d[0] = s[2];
        d[1] = s[1];
        d[2] = s[0];
    }
}

static void rgba2bgra_row(const uchar *restrict s, uchar *restrict d,
                          int width) {
    for (int x = 0; x < width; x++, s += 4, d += 4) {
        d[0] = s[2];
        d[1] = s[1];
        d[2] = s[0];
        d[3] = s[3];
    }
}

static void rgb2rgba_row(const uchar *restrict s, uchar *restrict d,
                         int width) {
    for (int x = 0; x < width; x++, s += 3, d += 4) {
        d[0] = s[0];
        d[1] = s[1];
        d[2] = s[2];
        d[3] = 255;
    }
}

static void rgba2rgb_row(const uchar *restrict s, uchar *restrict d,
                         int width) {
    for (int x = 0; x < width; x++, s += 4, d += 3) {
        d[0] = s[0];
        d[1] = s[1];
        d[2] = s[2];
    }
}

static void rgb2bgra_row(const uchar *restrict s, uchar *restrict d,
                         int width) {
    for (int x = 0; x < width; x++, s += 3, d += 4) {
        d[0] = s[2];
        d[1] = s[1];
        d[2] = s[0];
        d[3] = 255;
    }
}

static void rgba2bgr_row(const uchar *restrict s, uchar *restrict d,
                         int width) {
    for (int x = 0; x < width; x++, s += 4, d += 3) {
        d[0] = s[2];
        d[1] = s[1];
        d[2] = s[0];
    }
}

// HSV (H 0..179, S and V 0..255)
// RGB to HSV uses two reciprocal tables in Q12 so the row kernel has no
// division. sdiv_table[i] = round(255 * 4096 / i) and
// hdiv_table[i] = round(30 * 4096 / i), with entry 0 set to 0 (a zero
// divisor then yields 0 after the rounding term). The tables are static
// const; on AVR they occupy RAM (2 KB) unless the port moves them to
// PROGMEM.

static const uint32_t sdiv_table[256] = {
    0u,      1044480u, 522240u, 348160u, 261120u, 208896u, 174080u, 149211u,
    130560u, 116053u,  104448u, 94953u,  87040u,  80345u,  74606u,  69632u,
    65280u,  61440u,   58027u,  54973u,  52224u,  49737u,  47476u,  45412u,
    43520u,  41779u,   40172u,  38684u,  37303u,  36017u,  34816u,  33693u,
    32640u,  31651u,   30720u,  29842u,  29013u,  28229u,  27486u,  26782u,
    26112u,  25475u,   24869u,  24290u,  23738u,  23211u,  22706u,  22223u,
    21760u,  21316u,   20890u,  20480u,  20086u,  19707u,  19342u,  18991u,
    18651u,  18324u,   18008u,  17703u,  17408u,  17123u,  16846u,  16579u,
    16320u,  16069u,   15825u,  15589u,  15360u,  15137u,  14921u,  14711u,
    14507u,  14308u,   14115u,  13926u,  13743u,  13565u,  13391u,  13221u,
    13056u,  12895u,   12738u,  12584u,  12434u,  12288u,  12145u,  12006u,
    11869u,  11736u,   11605u,  11478u,  11353u,  11231u,  11111u,  10995u,
    10880u,  10768u,   10658u,  10550u,  10445u,  10341u,  10240u,  10141u,
    10043u,  9947u,    9854u,   9761u,   9671u,   9582u,   9495u,   9410u,
    9326u,   9243u,    9162u,   9082u,   9004u,   8927u,   8852u,   8777u,
    8704u,   8632u,    8561u,   8492u,   8423u,   8356u,   8290u,   8224u,
    8160u,   8097u,    8034u,   7973u,   7913u,   7853u,   7795u,   7737u,
    7680u,   7624u,    7569u,   7514u,   7461u,   7408u,   7355u,   7304u,
    7253u,   7203u,    7154u,   7105u,   7057u,   7010u,   6963u,   6917u,
    6872u,   6827u,    6782u,   6739u,   6695u,   6653u,   6611u,   6569u,
    6528u,   6487u,    6447u,   6408u,   6369u,   6330u,   6292u,   6254u,
    6217u,   6180u,    6144u,   6108u,   6073u,   6037u,   6003u,   5968u,
    5935u,   5901u,    5868u,   5835u,   5803u,   5771u,   5739u,   5708u,
    5677u,   5646u,    5615u,   5585u,   5556u,   5526u,   5497u,   5468u,
    5440u,   5412u,    5384u,   5356u,   5329u,   5302u,   5275u,   5249u,
    5222u,   5196u,    5171u,   5145u,   5120u,   5095u,   5070u,   5046u,
    5022u,   4998u,    4974u,   4950u,   4927u,   4904u,   4881u,   4858u,
    4836u,   4813u,    4791u,   4769u,   4748u,   4726u,   4705u,   4684u,
    4663u,   4642u,    4622u,   4601u,   4581u,   4561u,   4541u,   4522u,
    4502u,   4483u,    4464u,   4445u,   4426u,   4407u,   4389u,   4370u,
    4352u,   4334u,    4316u,   4298u,   4281u,   4263u,   4246u,   4229u,
    4212u,   4195u,    4178u,   4161u,   4145u,   4128u,   4112u,   4096u,
};

static const uint32_t hdiv_table[256] = {
    0u,     122880u, 61440u, 40960u, 30720u, 24576u, 20480u, 17554u, 15360u,
    13653u, 12288u,  11171u, 10240u, 9452u,  8777u,  8192u,  7680u,  7228u,
    6827u,  6467u,   6144u,  5851u,  5585u,  5343u,  5120u,  4915u,  4726u,
    4551u,  4389u,   4237u,  4096u,  3964u,  3840u,  3724u,  3614u,  3511u,
    3413u,  3321u,   3234u,  3151u,  3072u,  2997u,  2926u,  2858u,  2793u,
    2731u,  2671u,   2614u,  2560u,  2508u,  2458u,  2409u,  2363u,  2318u,
    2276u,  2234u,   2194u,  2156u,  2119u,  2083u,  2048u,  2014u,  1982u,
    1950u,  1920u,   1890u,  1862u,  1834u,  1807u,  1781u,  1755u,  1731u,
    1707u,  1683u,   1661u,  1638u,  1617u,  1596u,  1575u,  1555u,  1536u,
    1517u,  1499u,   1480u,  1463u,  1446u,  1429u,  1412u,  1396u,  1381u,
    1365u,  1350u,   1336u,  1321u,  1307u,  1293u,  1280u,  1267u,  1254u,
    1241u,  1229u,   1217u,  1205u,  1193u,  1182u,  1170u,  1159u,  1148u,
    1138u,  1127u,   1117u,  1107u,  1097u,  1087u,  1078u,  1069u,  1059u,
    1050u,  1041u,   1033u,  1024u,  1016u,  1007u,  999u,   991u,   983u,
    975u,   968u,    960u,   953u,   945u,   938u,   931u,   924u,   917u,
    910u,   904u,    897u,   890u,   884u,   878u,   871u,   865u,   859u,
    853u,   847u,    842u,   836u,   830u,   825u,   819u,   814u,   808u,
    803u,   798u,    793u,   788u,   783u,   778u,   773u,   768u,   763u,
    759u,   754u,    749u,   745u,   740u,   736u,   731u,   727u,   723u,
    719u,   714u,    710u,   706u,   702u,   698u,   694u,   690u,   686u,
    683u,   679u,    675u,   671u,   668u,   664u,   661u,   657u,   654u,
    650u,   647u,    643u,   640u,   637u,   633u,   630u,   627u,   624u,
    621u,   617u,    614u,   611u,   608u,   605u,   602u,   599u,   597u,
    594u,   591u,    588u,   585u,   582u,   580u,   577u,   574u,   572u,
    569u,   566u,    564u,   561u,   559u,   556u,   554u,   551u,   549u,
    546u,   544u,    541u,   539u,   537u,   534u,   532u,   530u,   527u,
    525u,   523u,    521u,   518u,   516u,   514u,   512u,   510u,   508u,
    506u,   504u,    502u,   500u,   497u,   495u,   493u,   492u,   490u,
    488u,   486u,    484u,   482u,
};

// Every intermediate is int32_t: hn is in [-255, 1275] and hdiv up to 122880,
// so hn * hdiv stays below 2^28. Negative values are never shifted.
static inline void rgb2hsv_px(int32_t r, int32_t g, int32_t b, uchar *d) {
    int32_t v    = r;
    int32_t minc = r;
    if (g > v) {
        v = g;
    }
    if (b > v) {
        v = b;
    }
    if (g < minc) {
        minc = g;
    }
    if (b < minc) {
        minc = b;
    }
    int32_t diff = v - minc;
    int32_t s    = (int32_t)(((uint32_t)diff * sdiv_table[v] + 2048u) >> 12);
    int32_t hn;
    if (v == r) {
        hn = g - b;
    } else if (v == g) {
        hn = b - r + 2 * diff;
    } else {
        hn = r - g + 4 * diff;
    }
    int32_t x = hn * (int32_t)hdiv_table[diff] + 2048;
    int32_t h;
    if (x >= 0) {
        h = x >> 12;
    } else {
        h = -((-x + 4095) >> 12);
    }
    if (h < 0) {
        h += 180;
    }
    d[0] = (uchar)h;
    d[1] = (uchar)s;
    d[2] = (uchar)v;
}

// R, G, B of each hue sector as indices into {V, P, Q or T}. A table instead
// of a switch: the sector changes from pixel to pixel.
static const uchar hsv_perm[6][3] = {{0, 2, 1}, {2, 0, 1}, {1, 0, 2},
                                     {1, 2, 0}, {2, 1, 0}, {0, 1, 2}};

static inline void hsv_store(const uchar o[3], uint32_t sec, int swap,
                             uchar *d) {
    const uchar *m  = hsv_perm[sec];
    d[swap ? 2 : 0] = o[m[0]];
    d[1]            = o[m[1]];
    d[swap ? 0 : 2] = o[m[2]];
}

#if IMPROC_HSV_FLOAT
// 1 - a * b rounded once to float32, for 0 <= a * b <= 1, without an FMA
// instruction: a * b is exact in double (24 + 24 bits), and TwoSum gives the
// rounding error of 1 - a * b, so 1 - a * b == d + err exactly. Rounding d to
// float32 is then wrong only when d is a float32 midpoint and err points past
// it; that case moves one float32 step. Every value here is positive, so a
// step is +-1 on the bit pattern.
static inline float hsv_one_minus_mul(float a, float b) {
    double p   = (double)a * (double)b;
    double d   = 1.0 - p;
    double err = (1.0 - d) - p;
    float  f   = (float)d;
    double gap = d - (double)f; // exact
    if (err != 0.0 && gap != 0.0) {
        uint32_t bits;
        memcpy(&bits, &f, sizeof bits);
        bits += gap > 0.0 ? 1u : (uint32_t)-1;
        float next;
        memcpy(&next, &bits, sizeof next);
        if (gap == ((double)next - (double)f) / 2.0 &&
            (err > 0.0) == (gap > 0.0)) {
            f = next;
        }
    }
    return f;
}

static inline void hsv2rgb_px(uint32_t h, uint32_t s, uint32_t v, int swap,
                              uchar *d) {
    const float inv = 1.0f / 255.0f;
    float       hh  = (float)h * (6.0f / 180.0f);
    uint32_t    k   = (uint32_t)hh; // trunc, hh >= 0
    float       f   = hh - (float)k;
    float       sf  = (float)s * inv;
    float       vf  = (float)v * inv;
    uint32_t    sec = k % 6u;
    // Odd sectors use q = V (1 - S F), even ones t = V (1 - S (1 - F)).
    float    x = (sec & 1u) ? f : 1.0f - f;
    uchar    o[3];
    o[0] = (uchar)(uint32_t)(vf * 255.0f);
    o[1] = (uchar)(uint32_t)(vf * (1.0f - sf) * 255.0f);
    o[2] = (uchar)(uint32_t)(vf * hsv_one_minus_mul(sf, x) * 255.0f);
    hsv_store(o, sec, swap, d);
}
#else
// HSV to RGB reproduces OpenCV's 8-bit vector code bit for bit, without float
// hardware. That code works in float32: S = s * (1/255), V = v * (1/255),
// H = h * (6/180), F = H - trunc(H), then V (1 - S), V (1 - S F) and
// V (1 - S (1 - F)), where 1 - S F and 1 - S (1 - F) are fused multiply-adds
// (one rounding). Each result is multiplied by 255 and truncated, not rounded.
//
// An F32 is the non-negative float32 value m * 2^e. f32_round rounds m * 2^e
// to 24 significant bits, ties to even, as float32 arithmetic does. All values
// here lie in [0, 256], so no overflow, underflow or subnormal case arises.
struct F32 {
    uint64_t m;
    int      e;
};

// Number of bits in m beyond the 24 a float32 keeps (0 when m < 2^24).
static inline int f32_excess_bits(uint64_t m) {
#if defined(__GNUC__) || defined(__clang__)
    return m >> 24 ? 40 - __builtin_clzll(m) : 0;
#else
    int n = 0;
    while ((m >> n) >= ((uint64_t)1 << 24)) {
        n++;
    }
    return n;
#endif
}

static struct F32 f32_round(uint64_t m, int e) {
    int n = f32_excess_bits(m);
    if (n > 0) {
        uint64_t q    = m >> n;
        uint64_t r    = m & (((uint64_t)1 << n) - 1u);
        uint64_t half = (uint64_t)1 << (n - 1);
        q += r > half || (r == half && (q & 1u));
        m = q;
        e += n;
    }
    return (struct F32){m, e};
}

static struct F32 f32_mul(struct F32 a, struct F32 b) {
    return f32_round(a.m * b.m, a.e + b.e); // both m <= 2^24
}

// 1 - a * b with a single rounding, for a * b <= 1. Here -59 <= a.e + b.e < 0
// (S has e in [-31, -23], F and 1 - F have e in [-28, -1]), so
// 2^-(a.e + b.e) fits in 64 bits.
static struct F32 f32_one_minus_mul(struct F32 a, struct F32 b) {
    int e = a.e + b.e;
    return f32_round(((uint64_t)1 << -e) - a.m * b.m, e);
}

static struct F32 f32_one_minus(struct F32 a) {
    return f32_one_minus_mul(a, (struct F32){1, 0});
}

// trunc(x * 255) for x in [0, 1.0000001]. A zero keeps the exponent of the
// subtraction that produced it, which can be below -63, so it returns early
// instead of shifting that far.
static uint32_t f32_trunc_255(struct F32 x) {
    struct F32 y = f32_round(x.m * 255u, x.e);
    if (y.m == 0 || y.e <= -64) {
        return 0;
    }
    return (uint32_t)(y.e >= 0 ? y.m << y.e : y.m >> -y.e);
}

#define F32_INV255 ((struct F32){8421505u, -31}) // float32(1/255)
#define F32_HSCALE ((struct F32){8947849u, -28}) // float32(6/180)

static inline void hsv2rgb_px(uint32_t h, uint32_t s, uint32_t v, int swap,
                              uchar *d) {
    struct F32 hh  = f32_round(h * F32_HSCALE.m, F32_HSCALE.e); // < 8.5, e < 0
    uint32_t   k   = (uint32_t)(hh.m >> -hh.e);
    struct F32 f   = {hh.m - ((uint64_t)k << -hh.e), hh.e}; // exact
    struct F32 sf  = f32_round(s * F32_INV255.m, F32_INV255.e);
    struct F32 vf  = f32_round(v * F32_INV255.m, F32_INV255.e);
    uint32_t   sec = k % 6u;
    struct F32 x   = (sec & 1u) ? f : f32_one_minus(f);
    uchar      o[3];
    o[0] = (uchar)f32_trunc_255(vf);
    o[1] = (uchar)f32_trunc_255(f32_mul(vf, f32_one_minus(sf)));
    o[2] = (uchar)f32_trunc_255(f32_mul(vf, f32_one_minus_mul(sf, x)));
    hsv_store(o, sec, swap, d);
}

#endif // IMPROC_HSV_FLOAT

static void rgb2hsv_row(const uchar *restrict s, uchar *restrict d, int width) {
    for (int x = 0; x < width; x++, s += 3, d += 3) {
        rgb2hsv_px(s[0], s[1], s[2], d);
    }
}

static void bgr2hsv_row(const uchar *restrict s, uchar *restrict d, int width) {
    for (int x = 0; x < width; x++, s += 3, d += 3) {
        rgb2hsv_px(s[2], s[1], s[0], d);
    }
}

static void hsv2rgb_row(const uchar *restrict s, uchar *restrict d, int width) {
    for (int x = 0; x < width; x++, s += 3, d += 3) {
        hsv2rgb_px(s[0], s[1], s[2], 0, d);
    }
}

static void hsv2bgr_row(const uchar *restrict s, uchar *restrict d, int width) {
    for (int x = 0; x < width; x++, s += 3, d += 3) {
        hsv2rgb_px(s[0], s[1], s[2], 1, d);
    }
}

// YUV (OpenCV's YCrCb; U is Cb and V is Cr, both centred on 128)
// OpenCV's Q14 integer constants. Forward: Y from the R2Y, G2Y, B2Y weights,
// then Cr = (R - Y) * 0.713 and Cb = (B - Y) * 0.564, both from the rounded Y.
// Inverse: R = Y + 1.403 Cr, G = Y - 0.714 Cr - 0.344 Cb, B = Y + 1.773 Cb.

#define YUV_SHIFT 14
#define YUV_HALF  ((int32_t)1 << (YUV_SHIFT - 1))

// round(x / 2^14) for a signed x, without shifting a negative value: the
// offset 2^24 is a multiple of 2^14 and keeps every x used here non-negative.
static inline int32_t yuv_descale(int32_t x) {
    const int32_t bias = (int32_t)1 << 24;
    return ((x + YUV_HALF + bias) >> YUV_SHIFT) - (bias >> YUV_SHIFT);
}

static inline uchar yuv_sat(int32_t x) {
    return (uchar)(x < 0 ? 0 : x > 255 ? 255 : x);
}

// |(R - Y) * 11682| <= 255 * 11682 < 2^22, so int32_t holds every term.
static inline void rgb2yuv_px(int32_t r, int32_t g, int32_t b, uchar *d) {
    int32_t y = yuv_descale(4899 * r + 9617 * g + 1868 * b);
    d[0]      = (uchar)y;
    d[1]      = yuv_sat(128 + yuv_descale((b - y) * 9241));
    d[2]      = yuv_sat(128 + yuv_descale((r - y) * 11682));
}

// |cr * 22987| <= 128 * 29049 < 2^22, so int32_t holds every term.
static inline void yuv2rgb_px(int32_t y, int32_t u, int32_t v, int swap,
                              uchar *d) {
    int32_t cb      = u - 128;
    int32_t cr      = v - 128;
    d[swap ? 2 : 0] = yuv_sat(y + yuv_descale(22987 * cr));
    d[1]            = yuv_sat(y + yuv_descale(-11698 * cr - 5636 * cb));
    d[swap ? 0 : 2] = yuv_sat(y + yuv_descale(29049 * cb));
}

static void rgb2yuv_row(const uchar *restrict s, uchar *restrict d, int width) {
    for (int x = 0; x < width; x++, s += 3, d += 3) {
        rgb2yuv_px(s[0], s[1], s[2], d);
    }
}

static void bgr2yuv_row(const uchar *restrict s, uchar *restrict d, int width) {
    for (int x = 0; x < width; x++, s += 3, d += 3) {
        rgb2yuv_px(s[2], s[1], s[0], d);
    }
}

static void yuv2rgb_row(const uchar *restrict s, uchar *restrict d, int width) {
    for (int x = 0; x < width; x++, s += 3, d += 3) {
        yuv2rgb_px(s[0], s[1], s[2], 0, d);
    }
}

static void yuv2bgr_row(const uchar *restrict s, uchar *restrict d, int width) {
    for (int x = 0; x < width; x++, s += 3, d += 3) {
        yuv2rgb_px(s[0], s[1], s[2], 1, d);
    }
}

// 565: the word t holds R in bits 15..11, G in 10..5, B in 4..0. t fits 16
// bits; the left shift may wrap a 16-bit unsigned, which the mask discards.

static inline unsigned word_le(const uchar *s) {
    return (unsigned)s[0] | ((unsigned)s[1] << 8);
}

static inline unsigned word_be(const uchar *s) {
    return ((unsigned)s[0] << 8) | (unsigned)s[1];
}

static inline uchar c565_b(unsigned t) { return (uchar)((t << 3) & 0xF8u); }

static inline uchar c565_g(unsigned t) { return (uchar)((t >> 3) & 0xFCu); }

static inline uchar c565_r(unsigned t) { return (uchar)((t >> 8) & 0xF8u); }

#define DEFINE_565_ROWS(suffix, read_word)                                     \
    static void bgr565##suffix##2bgr_row(const uchar *restrict s,              \
                                         uchar *restrict d, int width) {       \
        for (int x = 0; x < width; x++, s += 2, d += 3) {                      \
            unsigned t = read_word(s);                                         \
            d[0]       = c565_b(t);                                            \
            d[1]       = c565_g(t);                                            \
            d[2]       = c565_r(t);                                            \
        }                                                                      \
    }                                                                          \
    static void bgr565##suffix##2rgb_row(const uchar *restrict s,              \
                                         uchar *restrict d, int width) {       \
        for (int x = 0; x < width; x++, s += 2, d += 3) {                      \
            unsigned t = read_word(s);                                         \
            d[0]       = c565_r(t);                                            \
            d[1]       = c565_g(t);                                            \
            d[2]       = c565_b(t);                                            \
        }                                                                      \
    }                                                                          \
    static void bgr565##suffix##2gray_row(const uchar *restrict s,             \
                                          uchar *restrict d, int width) {      \
        for (int x = 0; x < width; x++, s += 2) {                              \
            unsigned t = read_word(s);                                         \
            d[x]       = luma(c565_r(t), c565_g(t), c565_b(t));                \
        }                                                                      \
    }

DEFINE_565_ROWS(le, word_le)
DEFINE_565_ROWS(be, word_be)

// YUYV: limited-range BT.601 with OpenCV's Q20 constants. The luma term is at
// most 239 * 1220542 = 291709538 and the largest chroma term is
// 128 * 2116026 = 270851328, so every sum stays below 2^30 and fits int32_t. A
// negative sum clamps to 0 before the shift, which gives the same result as
// OpenCV's arithmetic shift followed by saturation.
#define YUYV_CY   1220542
#define YUYV_CVR  1673527
#define YUYV_CVG  (-852492)
#define YUYV_CUG  (-409993)
#define YUYV_CUB  2116026
#define YUYV_HALF ((int32_t)1 << 19)

static inline uchar yuyv_sat(int32_t x) {
    if (x < 0) {
        return 0;
    }
    x >>= 20;
    return (uchar)(x > 255 ? 255 : x);
}

static inline void yuyv_px(int32_t y, int32_t ruv, int32_t guv, int32_t buv,
                           int swap, uchar *d) {
    int32_t yy      = (y > 16 ? y - 16 : 0) * (int32_t)YUYV_CY;
    d[swap ? 2 : 0] = yuyv_sat(yy + ruv);
    d[1]            = yuyv_sat(yy + guv);
    d[swap ? 0 : 2] = yuyv_sat(yy + buv);
}

static inline void yuyv_pair(const uchar *s, int swap, uchar *d) {
    int32_t u   = (int32_t)s[1] - 128;
    int32_t v   = (int32_t)s[3] - 128;
    int32_t ruv = YUYV_HALF + (int32_t)YUYV_CVR * v;
    int32_t guv = YUYV_HALF + (int32_t)YUYV_CVG * v + (int32_t)YUYV_CUG * u;
    int32_t buv = YUYV_HALF + (int32_t)YUYV_CUB * u;
    yuyv_px(s[0], ruv, guv, buv, swap, d);
    yuyv_px(s[2], ruv, guv, buv, swap, d + 3);
}

// width is even (cvt_color checks it).
static void yuyv2rgb_row(const uchar *restrict s, uchar *restrict d,
                         int width) {
    for (int x = 0; x < width; x += 2, s += 4, d += 6) {
        yuyv_pair(s, 0, d);
    }
}

static void yuyv2bgr_row(const uchar *restrict s, uchar *restrict d,
                         int width) {
    for (int x = 0; x < width; x += 2, s += 4, d += 6) {
        yuyv_pair(s, 1, d);
    }
}

static void yuyv2gray_row(const uchar *restrict s, uchar *restrict d,
                          int width) {
    for (int x = 0; x < width; x++, s += 2) {
        d[x] = s[0];
    }
}

typedef void (*color_row_fn)(const uchar *restrict, uchar *restrict, int);

struct ColorConversion {
    int          code;
    int          scn;
    int          dcn;
    color_row_fn kernel;
    int          even_cols; // 1 when the source packs two pixels per unit
};

static const struct ColorConversion conversions[] = {
    {COLOR_RGB2GRAY, 3, 1, rgb2gray_row, 0},
    {COLOR_BGR2GRAY, 3, 1, bgr2gray_row, 0},
    {COLOR_RGBA2GRAY, 4, 1, rgba2gray_row, 0},
    {COLOR_BGRA2GRAY, 4, 1, bgra2gray_row, 0},
    {COLOR_GRAY2RGB, 1, 3, gray2rgb_row, 0},
    {COLOR_GRAY2RGBA, 1, 4, gray2rgba_row, 0},
    {COLOR_RGB2BGR, 3, 3, rgb2bgr_row, 0},
    {COLOR_RGBA2BGRA, 4, 4, rgba2bgra_row, 0},
    {COLOR_RGB2RGBA, 3, 4, rgb2rgba_row, 0},
    {COLOR_RGBA2RGB, 4, 3, rgba2rgb_row, 0},
    {COLOR_RGB2BGRA, 3, 4, rgb2bgra_row, 0},
    {COLOR_RGBA2BGR, 4, 3, rgba2bgr_row, 0},
    {COLOR_RGB2HSV, 3, 3, rgb2hsv_row, 0},
    {COLOR_BGR2HSV, 3, 3, bgr2hsv_row, 0},
    {COLOR_HSV2RGB, 3, 3, hsv2rgb_row, 0},
    {COLOR_HSV2BGR, 3, 3, hsv2bgr_row, 0},
    {COLOR_RGB2YUV, 3, 3, rgb2yuv_row, 0},
    {COLOR_BGR2YUV, 3, 3, bgr2yuv_row, 0},
    {COLOR_YUV2RGB, 3, 3, yuv2rgb_row, 0},
    {COLOR_YUV2BGR, 3, 3, yuv2bgr_row, 0},
    {COLOR_BGR5652BGR, 2, 3, bgr565le2bgr_row, 0},
    {COLOR_BGR5652RGB, 2, 3, bgr565le2rgb_row, 0},
    {COLOR_BGR5652GRAY, 2, 1, bgr565le2gray_row, 0},
    {COLOR_BGR565BE2BGR, 2, 3, bgr565be2bgr_row, 0},
    {COLOR_BGR565BE2RGB, 2, 3, bgr565be2rgb_row, 0},
    {COLOR_BGR565BE2GRAY, 2, 1, bgr565be2gray_row, 0},
    {COLOR_YUV2RGB_YUYV, 2, 3, yuyv2rgb_row, 1},
    {COLOR_YUV2BGR_YUYV, 2, 3, yuyv2bgr_row, 1},
    {COLOR_YUV2GRAY_YUYV, 2, 1, yuyv2gray_row, 1},
};

void cvt_color(struct Mat *dst, const struct Mat *src, int code) {
    const struct ColorConversion *conv = NULL;
    for (size_t i = 0; i < sizeof(conversions) / sizeof(conversions[0]); i++) {
        if (conversions[i].code == code) {
            conv = &conversions[i];
            break;
        }
    }
    ASSERT(conv != NULL);
    ASSERT(dst != NULL && src != NULL);
    ASSERT(dst != src);
    ASSERT(mat_type(src) == (size_t)CV_MAKETYPE(CV_8U, conv->scn));
    ASSERT(mat_type(dst) == (size_t)CV_MAKETYPE(CV_8U, conv->dcn));
    ASSERT(src->rows == dst->rows && src->cols == dst->cols);
    ASSERT(!conv->even_cols || src->cols % 2 == 0);
    if (src->rows == 0 || src->cols == 0) {
        return;
    }
    ASSERT(dst->data != src->data);
    for (int y = 0; y < src->rows; y++) {
        conv->kernel(mat_row(src, y), mat_row(dst, y), src->cols);
    }
}

# OpenCV C implementation

以 C11 親手實作基本影像處理。支援 8 位元的 1、3、4 通道影像，以 RGB 為主；影像運算全部以整數定點運算完成（只有選用的 `convert_to` 用單精度浮點，且不使用 FMA），所以各平台的輸出逐 byte 相同。程式可編進樹莓派、Cortex-M 與 AVR 這類邊緣裝置（見下方「邊緣裝置建置」）。

## 用途

在微控制器上開發時，常常找不到能直接編譯上去的影像處理函式庫，只能自己手寫。OpenCV 太大，也依賴作業系統。這個專案提供一組小而完整的影像處理函式，目標是「在受限裝置上可重現、可驗證」：

- **可重現：** 全部用整數定點運算，同一份 Kodak 雜湊清單在 Windows（LLVM-MinGW）、Linux x64（GCC）與 aarch64（`qemu-aarch64`）上都通過，輸出逐 byte 相同。同一張圖在 PC 上算出的結果，可以直接當成裝置上的期望值。Cortex-M 與 AVR 目前只驗證編譯與大小，還沒有在實機上比對輸出。
- **受限裝置友善：** 核心只依賴 libc（開啟浮點時另加 libm），不需要檔案系統、執行緒或作業系統。暫存記憶體只與列寬和 `ksize` 有關，大小都有文件記載。影像緩衝區可以由呼叫端提供，不必用 `alloc_mat`；需要暫存的運算則可接上自己的配置器。
- **可驗證：** 每個運算都有手算的小矩陣測試、與 naive 實作的比對，以及在 24 張 Kodak 影像上與 OpenCV 的逐運算比對（最大差異與容許誤差見 [docs/kodak-results.md](docs/kodak-results.md)）。

適合的情境：相機模組上的前處理（灰階、模糊、二值化、縮圖）、簡單的物體偵測前置步驟，以及需要可當參考答案的影像處理實作。這也是一個學習與作品集專案，所有演算法都是自己實作，API 仍可能調整。

## 安裝

### 需要的工具

| 平台 | 編譯器 | 其他 |
|---|---|---|
| Windows x64 | [LLVM-MinGW](https://github.com/mstorsjo/llvm-mingw)（MSVC 不支援） | CMake 3.21 以上 |
| Linux x64 | GCC | CMake 3.21 以上 |
| macOS arm64 | Clang（Xcode Command Line Tools） | CMake 3.21 以上 |

下方指令使用 Ninja（建議安裝）；沒有 Ninja 時拿掉 `-G Ninja`，CMake 會改用平台預設的產生器。與 OpenCV 比對時另外需要 Python 3、`opencv-python` 與 `numpy`；只使用函式庫則不需要。

### 下載與建置

```sh
git clone https://github.com/YuMingCorn/opencv_c.git
cd opencv_c
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure   # 沒有 Kodak 資料集時，Kodak 測試會顯示為 Skipped
```

建置會產生兩個靜態函式庫：

- `improc_core`：可攜核心，所有影像運算都在這裡。不含 stdio、pthread 與檔案 IO，可以交叉編譯到微控制器。
- `improc`：`improc_core` 加上 PNG／JPG 讀寫（`mat_io.h`）。PC 與樹莓派上用這個。

### 放進自己的 CMake 專案

把這個 repo 放進專案（例如 `external/opencv_c`，可用 `git submodule add`），再在 `CMakeLists.txt` 加上：

```cmake
# PC 或樹莓派：需要讀寫圖檔
add_subdirectory(external/opencv_c)
target_link_libraries(myapp PRIVATE improc)
```

```cmake
# 微控制器或只要影像運算：只建 improc_core，不建圖檔 IO、demo 與測試
set(IMPROC_CORE_ONLY ON CACHE BOOL "" FORCE)
add_subdirectory(external/opencv_c)
target_link_libraries(myapp PRIVATE improc_core)
```

使用 `cmake/toolchains/` 內的裸機工具鏈檔時，`IMPROC_CORE_ONLY` 預設就是 ON。include 路徑會自動帶入。程式中寫 `#include "improc.h"` 即可使用全部函式。

## 功能

| 模組 | 函式 | 說明 |
|---|---|---|
| `color.h` | `cvt_color` | GRAY、RGB、BGR、RGBA、BGRA 互轉；灰階用 `(77R + 150G + 29B + 128) >> 8` |
| `color.h` | `cvt_color`（HSV、YUV） | RGB／BGR 與 HSV（H 0～179）互轉；RGB／BGR 與 YUV（JFIF 全範圍 BT.601）互轉 |
| `pixel.h` | `threshold` | BINARY、BINARY_INV、TRUNC、TOZERO、TOZERO_INV |
| `pixel.h` | `mat_lut` | 256 項查表 |
| `pixel.h` | `convert_scale`、`convert_scale_abs` | `src * alpha_q8 / 256 + beta`，結果飽和到 0～255 |
| `filter.h` | `box_filter`、`gaussian_blur`、`sobel` | 邊界一律 BORDER_REFLECT_101；以逐列滾動的環形緩衝區運作，暫存記憶體只與 `ksize` 和列寬有關；Sobel 輸入灰階、輸出 `CV_16SC1` |
| `morph.h` | `erode`、`dilate`、`morph_open`、`morph_close` | 矩形結構元素，`ksize` 為 1～255 的奇數；視窗超出影像的部分不計入 |
| `hist.h` | `calc_hist`、`equalize_hist` | 256 階直方圖與直方圖等化 |
| `hist.h` | `otsu_threshold`、`otsu_threshold_hist` | Otsu 門檻；判準以精確的整數與分數比較，沒有浮點捨入造成的平手 |
| `geometry.h` | `flip` | 水平、垂直、雙向翻轉 |
| `geometry.h` | `resize` | `INTER_NEAREST`、`INTER_LINEAR`（半像素中心）與 `INTER_AREA`（整數面積權重，放大縮小皆可） |
| `convert_f32.h` | `convert_to` | `CV_8UCn` 與 `CV_32FCn` 互轉並套用 `alpha`、`beta`；需要 `IMPROC_ENABLE_FLOAT` |
| `mat.h` | `alloc_mat`、`mat_init` | `alloc_mat` 從堆積配置；`mat_init` 包裝呼叫端的緩衝區，不配置記憶體 |
| `mat_io.h` | `alloc_mat_from_file`、`mat_write` | 讀寫 PNG 與 JPG，依副檔名選格式（只在 `improc` 函式庫，不在 `improc_core`） |

## 範例程式碼

所有函式都採「呼叫端配置 dst」的方式：先用 `alloc_mat` 配置好輸出，再把它傳進運算。尺寸或型別不符時會以 ASSERT 中止；讀檔失敗這類外部錯誤則回傳 `NULL` 或錯誤碼。

### 1. 灰階、模糊與 Otsu 二值化

```c
#include "improc.h"

int main(void) {
    struct Mat *src = alloc_mat_from_file("input.png");   // 讀入後為 RGB 順序
    if (src == NULL) {
        return 1;                                           // 檔案不存在或格式不支援
    }
    struct Mat *gray = alloc_mat(src->rows, src->cols, CV_8UC1);
    struct Mat *blur = alloc_mat(src->rows, src->cols, CV_8UC1);
    struct Mat *bin  = alloc_mat(src->rows, src->cols, CV_8UC1);

    cvt_color(gray, src, COLOR_RGB2GRAY);
    gaussian_blur(blur, gray, 5);                           // 5x5 高斯，去除雜訊
    uchar t = otsu_threshold(blur);                         // 自動決定門檻
    threshold(bin, blur, t, 255, THRESH_BINARY);
    mat_write("binary.png", bin);                           // 依副檔名存成 PNG 或 JPG

    free_mat(&bin);
    free_mat(&blur);
    free_mat(&gray);
    free_mat(&src);
    return 0;
}
```

### 2. 邊緣偵測（Sobel）

`sobel` 輸出帶正負號的 `CV_16SC1`，再用 `convert_scale_abs` 取絕對值轉回 8 位元：

```c
struct Mat *gx    = alloc_mat(blur->rows, blur->cols, CV_16SC1);
struct Mat *edges = alloc_mat(blur->rows, blur->cols, CV_8UC1);
sobel(gx, blur, 1, 0);                    // x 方向導數；(0, 1) 為 y 方向
convert_scale_abs(edges, gx, 256, 0);     // alpha 以 1/256 為單位，256 代表 1.0
mat_write("edges_x.png", edges);
```

### 3. 縮圖

```c
struct Mat *small = alloc_mat(src->rows / 2, src->cols / 2, CV_8UC3);
resize(small, src, INTER_AREA);           // 縮小用 INTER_AREA；也有 INTER_NEAREST、INTER_LINEAR
mat_write("small.jpg", small);
```

### 4. 不使用堆積（微控制器）

在沒有 malloc 的環境，用 `mat_init` 把自己的緩衝區包成 `Mat`。只連結 `improc_core`，不需要檔案系統：

```c
#include "improc.h"

static uchar frame_buf[120 * 160];        // 例如相機給的 160x120 灰階畫面
static uchar mask_buf[120 * 160];

void process_frame(void) {
    struct Mat frame, mask;
    mat_init(&frame, 120, 160, CV_8UC1, frame_buf, sizeof frame_buf);
    mat_init(&mask, 120, 160, CV_8UC1, mask_buf, sizeof mask_buf);

    uint32_t hist[256];
    calc_hist(hist, &frame);
    uchar t = otsu_threshold_hist(hist);
    threshold(&mask, &frame, t, 255, THRESH_BINARY);
}
```

`cvt_color`、`threshold`、`mat_lut`、`convert_scale`、`flip`、`calc_hist`、`otsu_threshold_hist` 不需要暫存記憶體。濾波、形態學、`resize`、`equalize_hist` 與 `otsu_threshold` 會向 `fast_malloc` 要一小塊暫存，沒有 malloc 時要用 `improc_set_allocator` 提供自己的配置器（例如固定大小的記憶體池）。各運算的暫存大小見 [docs/porting.md](docs/porting.md)。

完整的可執行範例是 `main.c`（建置後的 `demo`），它對一張圖依序跑過所有運算並各存一張 PNG，見下方「執行」。

## 建置與測試

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

常用 CMake 選項（完整說明見 [docs/porting.md](docs/porting.md)）：

| 選項 | 預設 | 作用 |
|---|---|---|
| `IMPROC_CORE_ONLY` | 裸機為 ON | 只建 `improc_core`，不建圖檔 IO、demo、benchmark 與測試 |
| `IMPROC_NO_STDIO` | 裸機為 ON | 核心不引入 `<stdio.h>` |
| `IMPROC_NO_MALLOC` | OFF | 沒有預設配置器，改用 `mat_init` |
| `IMPROC_ENABLE_FLOAT` | ON | 編譯 `convert_to` 等浮點程式 |
| `IMPROC_ASSERT_TERSE` | OFF | ASSERT 只傳行號，省下字串常數 |
| `IMPROC_WERROR` | OFF | 警告視為錯誤 |
| `IMPROC_KODAK_DIR` | `datasets/kodak` | Kodak 資料集位置 |

### Kodak 測試

`test_kodak_*` 用 24 張 Kodak 影像跑每個運算，把輸出的 FNV-1a 雜湊與 `tests/golden/kodak_*.txt` 比對，確認各平台逐 byte 相同。資料集不在版本庫內（`datasets/` 已被 git 忽略）。請自行下載 24 張 `kodim01.png`～`kodim24.png` 放進 `datasets/kodak`，來源為 <https://r0k.us/graphics/kodak/>。沒有資料集時這些測試會顯示為 Skipped，其餘測試照常執行。也可以用環境變數 `IMPROC_KODAK_DIR` 或 CMake 選項 `-DIMPROC_KODAK_DIR=<目錄>` 指定其他位置。

### 與 OpenCV 比對

比對腳本需要 Python 3、`opencv-python` 與 `numpy`。先用各測試的 `--dump` 輸出影像，再讓腳本逐運算算出與 OpenCV 的最大差異：

```sh
for m in phase1 morph hist color2 area f32; do
  build/test_kodak_$m --dir datasets/kodak --manifest tests/golden/kodak_$m.txt --dump dump
done
python3 tools/compare_opencv.py --dump dump
```

每個運算有各自的容許誤差，超過就以非零狀態結束。結果與解讀見 [docs/kodak-results.md](docs/kodak-results.md)。

## 邊緣裝置建置

`improc_core` 只依賴 libc 的 `memcpy`、`abort` 這類函式，可用 `cmake/toolchains/` 內的工具鏈檔交叉編譯：

```sh
cmake -S . -B build-m4f -G Ninja --toolchain cmake/toolchains/cortex-m4f.cmake -DCMAKE_BUILD_TYPE=MinSizeRel
cmake --build build-m4f --target improc_core
bash tools/check_cross.sh      # 對所有已安裝的編譯器編譯並檢查未定義符號
```

支援樹莓派（aarch64、armhf）、Cortex-M0+／M4F／M7 與 AVR（只驗證編譯與大小）。旗標表、函式庫大小、各運算的暫存記憶體與沒有 malloc 時的用法見 [docs/porting.md](docs/porting.md)。

## 執行

```sh
cmake -S . -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-release

# 不帶參數時使用內建的合成測試圖；對每種運算各輸出一張 <prefix><name>.png（預設 prefix 為 out_）
./build-release/demo [input-image] [output-prefix]

# 量測 RGB 轉灰階的耗時
./build-release/bench_gray
```

## 架構圖

模組關係、`struct Mat` 資料模型與各演算法流程的圖解見 [docs/README.md](docs/README.md)。

## 授權

MIT，見 [LICENSE](LICENSE)。`core/stb_image.h` 與 `core/stb_image_write.h` 是第三方檔案，沿用各自檔內的授權（只用於 `improc` 的圖檔讀寫，不在 `improc_core` 內）。

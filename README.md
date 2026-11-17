# improc：給邊緣裝置用的 C11 影像處理函式庫

[![CI](https://github.com/YuMingCorn/OpenCV_For_Edge_Device/actions/workflows/ci.yml/badge.svg)](https://github.com/YuMingCorn/OpenCV_For_Edge_Device/actions/workflows/ci.yml)

improc 是一組用 C11 寫成的基本影像處理函式：色彩轉換、二值化、濾波、形態學、直方圖、幾何轉換。它不依賴 OpenCV 或其他影像處理函式庫，可以編進樹莓派、Cortex-M、ESP32 與 AVR。影像運算全部用整數定點運算，所以每個平台的輸出逐 byte 相同。

> **這個函式庫由 AI 撰寫。** 程式碼、測試與文件由 AI 程式設計助理（Anthropic 的 Claude，透過 Claude Code）寫成。維護者負責決定方向、審查每一個 pull request，並決定是否合併。正確性靠自動測試與和 OpenCV 的逐運算比對來確認，詳見[由 AI 開發](#由-ai-開發)。

## 目錄

- [這個函式庫是什麼](#這個函式庫是什麼)
- [由 AI 開發](#由-ai-開發)
- [安裝](#安裝)
  - [支援的平台](#支援的平台)
  - [Windows x64](#windows-x64)
  - [Linux x64](#linux-x64)
  - [macOS arm64](#macos-arm64)
  - [樹莓派](#樹莓派)
  - [微控制器（Cortex-M、ESP32、AVR）](#微控制器cortex-mesp32avr)
  - [Arduino IDE 與 PlatformIO](#arduino-ide-與-platformio)
  - [建置選項](#建置選項)
- [快速開始](#快速開始)
- [範例程式碼](#範例程式碼)
- [功能](#功能)
- [測試與驗證](#測試與驗證)
- [文件](#文件)
- [授權](#授權)

## 這個函式庫是什麼

在微控制器上開發時，常常找不到能直接編譯上去的影像處理函式庫。OpenCV 太大，也需要作業系統。improc 提供一組小而完整的函式，目標是在受限裝置上「可重現、可驗證」：

- **可重現。** 影像運算只用整數定點運算（只有選用的 `convert_to` 用單精度浮點，且不使用 FMA）。同一份 Kodak 雜湊清單在 Windows、Linux x64 與 aarch64 上都通過，輸出逐 byte 相同。所以在 PC 上算出的結果，可以直接當成裝置上的期望值。
- **適合受限裝置。** 核心只依賴 libc（開啟浮點時另加 libm），不需要檔案系統、執行緒或作業系統。暫存記憶體只與列寬和 `ksize` 有關，大小寫在文件裡。影像緩衝區可以由呼叫端提供；需要暫存的運算可以接上自己的配置器。
- **可驗證。** 每個運算都有手算的小矩陣測試、與簡單參考實作的比對，以及在 24 張 Kodak 影像上與 OpenCV 的逐運算比對。

適合的用途：相機模組上的前處理（灰階、模糊、二值化、縮圖）、簡單物體偵測的前置步驟，以及需要一份可當參考答案的影像處理實作。API 仍可能調整。

建置會產生兩個靜態函式庫：

| 函式庫 | 內容 | 用在哪裡 |
|---|---|---|
| `improc_core` | 全部影像運算。不含 stdio、pthread 與檔案 IO | 微控制器、Arduino，或任何只需要運算的程式 |
| `improc` | `improc_core` 加上 PNG／JPG 讀寫（`mat_io.h`，使用 stb） | PC 與樹莓派 |

## 由 AI 開發

這個專案的程式碼不是人寫的。分工如下：

| 誰 | 做什麼 |
|---|---|
| AI（Claude，透過 Claude Code） | 撰寫演算法、測試、建置腳本、CI 設定與文件；查證 OpenCV 的行為；執行測試與交叉編譯；開 pull request |
| 維護者（[@YuMingCorn](https://github.com/YuMingCorn)） | 決定要做什麼與設計取捨；審查每一個 pull request；決定是否合併 |

AI 寫的程式可能有人和 AI 都沒發現的錯誤，所以正確性不靠「看起來對」，而是靠可以重跑的檢查：

- 每個運算都有手算期望值的小矩陣測試，並在寬度 0～65 上與測試內的簡單參考實作比對。
- 24 張 Kodak 影像的輸出雜湊存在 `tests/golden/`。Windows（LLVM-MinGW）、Linux x64（GCC）與 aarch64（在 QEMU 上執行）都要得到相同的雜湊。
- `tools/compare_opencv.py` 把每個運算的輸出與 OpenCV 比對，並檢查各自的容許誤差。多數運算與 OpenCV 逐 byte 相同，其餘運算的最大差異是 1。結果見 [docs/kodak-results.md](docs/kodak-results.md)。
- CI 在每個 pull request 上跑單元測試（Windows、Linux、macOS 與 QEMU 上的 aarch64），並交叉編譯到樹莓派、Cortex-M、ESP32 與 AVR。Kodak 資料集不在版本庫內，所以 Kodak 雜湊與 OpenCV 比對不在 CI 上跑，而是在合併前於本機執行。

還沒有驗證的部分：macOS 沒有跑過 Kodak 雜湊比對；微控制器（Cortex-M、ESP32、AVR）只驗證過編譯與程式大小，還沒有在實機上比對輸出；樹莓派的測試是在 QEMU 模擬器上跑的，還沒有在實機上跑。用在重要的產品前，請在自己的硬體上跑一次測試。

## 安裝

### 支援的平台

| 平台 | 編譯器 | 驗證方式 |
|---|---|---|
| Windows x64 | [LLVM-MinGW](https://github.com/mstorsjo/llvm-mingw)（Clang）。不支援 MSVC | CI 建置並跑單元測試 |
| Linux x64 | GCC | CI 建置並跑單元測試 |
| macOS arm64 | Apple Clang（Xcode Command Line Tools） | CI 建置並跑單元測試 |
| 樹莓派 64 位元（aarch64） | GCC | CI 交叉編譯，並在 `qemu-aarch64` 上跑單元測試 |
| 樹莓派 32 位元（armhf） | GCC | CI 交叉編譯 `improc_core` |
| Cortex-M0+／M4F／M7 | `arm-none-eabi-gcc` | CI 編譯 `improc_core` 並檢查大小與未定義符號 |
| ESP32 | `xtensa-esp32-elf-gcc`（esp32 Arduino core 附帶） | CI 編譯 `improc_core`；Arduino 範例在本機用 arduino-cli 編譯過 |
| AVR（ATmega328P） | `avr-gcc` | CI 編譯 `improc_core` |

每個平台都需要 CMake 3.21 以上。下面的指令使用 [Ninja](https://ninja-build.org/)；沒有 Ninja 時拿掉 `-G Ninja`，CMake 會改用平台預設的產生器。Arduino IDE 不需要這兩者。

先取得原始碼：

```sh
git clone https://github.com/YuMingCorn/OpenCV_For_Edge_Device.git opencv_c
cd opencv_c
```

### Windows x64

1. 用 winget 安裝編譯器、CMake、Ninja 與 Git。在 PowerShell 執行：

   ```powershell
   winget install --id MartinStorsjo.LLVM-MinGW.UCRT -e
   winget install --id Kitware.CMake -e
   winget install --id Ninja-build.Ninja -e
   winget install --id Git.Git -e
   ```

2. 關掉 PowerShell 再重新開啟，讓新的 `PATH` 生效。確認三個工具都找得到：

   ```powershell
   clang --version
   cmake --version
   ninja --version
   ```

3. 建置並測試。`$env:CC = "clang"` 讓 CMake 使用 LLVM-MinGW，而不是電腦上其他的編譯器：

   ```powershell
   $env:CC = "clang"
   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
   cmake --build build
   ctest --test-dir build --output-on-failure
   ```

4. 執行 demo。它對內建的測試圖跑過每一種運算，並把結果存成 `out_<名稱>.png`：

   ```powershell
   .\build\demo.exe
   ```

### Linux x64

以 Ubuntu 與 Debian 為例：

```sh
sudo apt update
sudo apt install build-essential cmake ninja-build git

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
./build/demo
```

### macOS arm64

用 [Homebrew](https://brew.sh/) 安裝 CMake 與 Ninja：

```sh
xcode-select --install          # 安裝 Apple Clang；已安裝時會顯示錯誤，可以忽略
brew install cmake ninja

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
./build/demo
```

### 樹莓派

有兩種做法：直接在樹莓派上建置，或在 Linux PC 上交叉編譯再複製過去。

**直接在樹莓派上建置。** Raspberry Pi OS 的步驟與 [Linux x64](#linux-x64) 相同。舊版系統（例如 Bullseye）用 apt 裝的 CMake 低於 3.21，這時改用 `pip install cmake` 安裝新版：

```sh
sudo apt install build-essential cmake ninja-build git
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

**在 Linux PC 上交叉編譯 64 位元版本（aarch64）。** 測試會透過 `qemu-aarch64` 在 PC 上執行：

```sh
sudo apt install gcc-aarch64-linux-gnu qemu-user

cmake -S . -B build-aarch64 -G Ninja -DCMAKE_BUILD_TYPE=Release \
    --toolchain cmake/toolchains/aarch64-linux-gnu.cmake
cmake --build build-aarch64
ctest --test-dir build-aarch64 --output-on-failure
```

建出來的程式（例如 `build-aarch64/demo`）可以用 `scp` 複製到 64 位元的 Raspberry Pi OS 上執行。

**交叉編譯 32 位元版本（armhf）。** 這個工具鏈檔沒有設定模擬器，所以 `ctest` 不能在 PC 上跑，要把程式複製到樹莓派上執行：

```sh
sudo apt install gcc-arm-linux-gnueabihf

cmake -S . -B build-armhf -G Ninja -DCMAKE_BUILD_TYPE=Release \
    --toolchain cmake/toolchains/armhf-linux-gnueabihf.cmake
cmake --build build-armhf
```

### 微控制器（Cortex-M、ESP32、AVR）

微控制器只建 `improc_core`。`cmake/toolchains/` 內有各目標的工具鏈檔，它們會自動開啟 `IMPROC_CORE_ONLY`，所以不會建圖檔 IO、demo 與測試。

1. 安裝編譯器（Ubuntu 與 Debian）：

   ```sh
   sudo apt install gcc-arm-none-eabi libnewlib-arm-none-eabi   # Cortex-M
   sudo apt install gcc-avr avr-libc                            # AVR
   ```

   ESP32 的編譯器使用 esp32 Arduino core 附帶的版本。用 Arduino IDE 或 arduino-cli 安裝 esp32 core 之後，把它加進 `PATH`：

   ```sh
   export PATH="$(ls -d ~/.arduino15/packages/esp32/tools/esp-x32/*/bin | tail -1):$PATH"
   ```

2. 用對應的工具鏈檔建置。以 Cortex-M4F 為例：

   ```sh
   cmake -S . -B build-m4f -G Ninja -DCMAKE_BUILD_TYPE=MinSizeRel \
       --toolchain cmake/toolchains/cortex-m4f.cmake
   cmake --build build-m4f --target improc_core
   arm-none-eabi-size -t build-m4f/libimproc_core.a
   ```

   其他目標換成 `cortex-m0plus.cmake`、`cortex-m7.cmake`、`esp32.cmake` 或 `avr-atmega328p.cmake`。結果是 `libimproc_core.a`，連結進自己的韌體即可，標頭檔在 `core/`。

3. 一次檢查所有已安裝的交叉編譯器：

   ```sh
   bash tools/check_cross.sh
   ```

   它對每個找得到的編譯器建出 `improc_core`、印出大小，並檢查有沒有用到微控制器上不該有的函式（例如 `printf`、`malloc`）。

**不用 CMake 的專案（Makefile、STM32CubeIDE、Keil）。** 把 `core/` 內除了 `mat_io.c` 以外的 `.c` 檔加進專案，加上 include 路徑 `core/` 與下面的巨集。例如用 `arm-none-eabi-gcc` 直接編譯：

```sh
FLAGS="-std=c11 -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard -Os \
       -ffunction-sections -fdata-sections -ffp-contract=off -Icore \
       -DIMPROC_CORE_ONLY=1 -DIMPROC_NO_STDIO=1 -DIMPROC_HAS_FLOAT=1"
for f in color pixel filter geometry mat utils morph hist features2d draw convert_f32; do
    arm-none-eabi-gcc $FLAGS -c core/$f.c
done
arm-none-eabi-ar rcs libimproc_core.a *.o
```

沒有浮點的目標改用 `-DIMPROC_HAS_FLOAT=0`，並拿掉 `convert_f32`。各巨集的意義見 [docs/porting.md](docs/porting.md)。

### Arduino IDE 與 PlatformIO

這個 repo 同時也是一個 Arduino 函式庫，名稱是 `improc`。它只包含 `improc_core`，不包含圖檔讀寫。

- **Arduino IDE：** 在 GitHub 下載 ZIP，再選「草稿碼 → 匯入程式庫 → 加入 .ZIP 程式庫」。
- **PlatformIO：** 在 `platformio.ini` 加上 `lib_deps = https://github.com/YuMingCorn/OpenCV_For_Edge_Device.git`。要指定分支時，在網址後面加 `#<分支>`。

在程式中寫 `#include <improc.h>` 即可使用。C 與 C++（`.ino`）都可以。內附兩個範例：

| 範例 | 板子 | 內容 |
|---|---|---|
| `examples/Synthetic` | RAM 約 128 KB 以上的板子（ESP32、ESP32-S3、Cortex-M） | 用內建的合成圖跑「灰階 → Gaussian 5x5 → Otsu」，印出每一步的時間與雜湊。雜湊和 PC 相同時印出 `PASS`，代表這塊板子的輸出和 PC 逐 byte 相同 |
| `examples/CameraOtsu` | AI-Thinker ESP32-CAM | 從相機抓 QVGA 灰階畫面，做 Gaussian 與 Otsu 二值化，印出時間、門檻值與白色像素比例 |

> 目前狀態：兩個範例都已用 arduino-cli（esp32 core 3.3.12）為 ESP32-CAM 編譯成功，`Synthetic` 也為 ESP32-S3 編譯成功。兩者都還沒在實機上執行，PlatformIO 也還沒實際編譯過。Arduino Uno 的 RAM 不夠執行 `Synthetic`。

Arduino 不能傳 `-D` 旗標，所以由 `src/improc_config.h` 提供預設值：一律定義 `IMPROC_CORE_ONLY`；AVR 關閉浮點，並開啟 `IMPROC_NO_STDIO` 與 `IMPROC_ASSERT_TERSE`；其他架構開啟浮點。要改設定時，在建置旗標中定義同名巨集，例如 PlatformIO 的 `build_flags = -DIMPROC_ASSERT_TERSE=1`。`src/` 內的 `.c` 檔只 include `core/` 的原始檔，所以 Arduino 與 CMake 編譯的是同一份程式碼。

沒有 arduino-cli 時，可以用 `bash tools/check_arduino_src.sh` 檢查這個結構。用 Git Bash 時請加 `HOST_CC=cc HOST_CXX=c++`。

### 建置選項

在 `cmake -S . -B build` 後面加 `-D<選項>=ON` 或 `OFF`。完整說明見 [docs/porting.md](docs/porting.md)。

| 選項 | 預設 | 作用 |
|---|---|---|
| `CMAKE_BUILD_TYPE` | 無 | `Debug`、`Release`，微控制器用 `MinSizeRel`。效能數字一律來自 `Release` |
| `IMPROC_CORE_ONLY` | 裸機為 ON | 只建 `improc_core`，不建圖檔 IO、demo、benchmark 與測試 |
| `IMPROC_NO_STDIO` | 裸機為 ON | 核心不引入 `<stdio.h>` |
| `IMPROC_NO_MALLOC` | OFF | 沒有預設配置器，影像改用 `mat_init` 包裝自己的緩衝區 |
| `IMPROC_ENABLE_FLOAT` | ON | 編譯 `convert_to` 等浮點程式 |
| `IMPROC_ASSERT_TERSE` | OFF | ASSERT 只傳行號，省下字串常數 |
| `IMPROC_WERROR` | OFF | 警告視為錯誤 |
| `IMPROC_KODAK_DIR` | `datasets/kodak` | Kodak 資料集位置 |

## 快速開始

`samples/quickstart/` 是一個獨立的小專案，示範怎麼在自己的程式裡使用 improc。它讀一張圖（沒有給圖時使用內建的測試圖），做灰階、5x5 Gaussian 模糊與 Otsu 二值化，再存成 `binary.png`。

`samples/quickstart/CMakeLists.txt`：

```cmake
cmake_minimum_required(VERSION 3.21)
project(quickstart LANGUAGES C)

# Copy this folder anywhere and point IMPROC_DIR at the opencv_c checkout.
set(IMPROC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH
    "opencv_c source directory")
add_subdirectory(${IMPROC_DIR} improc)

add_executable(quickstart main.c)
target_link_libraries(quickstart PRIVATE improc)
```

`samples/quickstart/main.c`：

```c
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
```

在 repo 的根目錄建置並執行。Windows（PowerShell）：

```powershell
$env:CC = "clang"
cmake -S samples/quickstart -B build-quickstart -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-quickstart
.\build-quickstart\quickstart.exe
```

Linux、macOS 與樹莓派：

```sh
cmake -S samples/quickstart -B build-quickstart -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-quickstart
./build-quickstart/quickstart
```

輸出應該是：

```text
320x240, Otsu threshold 127, wrote binary.png
```

給它一張自己的圖：`./build-quickstart/quickstart photo.jpg result.png`。

**用在自己的專案。** 把 `samples/quickstart` 複製到任何地方，建置時用 `IMPROC_DIR` 指向 opencv_c 的位置：

```sh
cmake -S myapp -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DIMPROC_DIR=/path/to/opencv_c
```

也可以把這個 repo 當成 git submodule 放進專案，例如 `git submodule add https://github.com/YuMingCorn/OpenCV_For_Edge_Device.git external/opencv_c`，再在 `CMakeLists.txt` 寫 `add_subdirectory(external/opencv_c)`。用 `add_subdirectory` 引入時，只會建出 `improc_core` 與 `improc` 兩個函式庫，不會建 demo 與測試。只需要影像運算（例如微控制器）時，在 `add_subdirectory` 之前加 `set(IMPROC_CORE_ONLY ON CACHE BOOL "" FORCE)`，並連結 `improc_core`。

## 範例程式碼

所有函式都由呼叫端配置輸出：先用 `alloc_mat` 配置好 dst，再把它傳進運算。尺寸或型別不符時，函式會以 ASSERT 中止；讀檔失敗這類外部錯誤則回傳 `NULL` 或錯誤碼。

### 邊緣偵測（Sobel）

`sobel` 輸出帶正負號的 `CV_16SC1`，再用 `convert_scale_abs` 取絕對值轉回 8 位元：

```c
struct Mat *gx    = alloc_mat(blur->rows, blur->cols, CV_16SC1);
struct Mat *edges = alloc_mat(blur->rows, blur->cols, CV_8UC1);
sobel(gx, blur, 1, 0);                    // x 方向導數；(0, 1) 為 y 方向
convert_scale_abs(edges, gx, 256, 0);     // alpha 以 1/256 為單位，256 代表 1.0
mat_write("edges_x.png", edges);
```

### 縮圖

```c
struct Mat *small = alloc_mat(src->rows / 2, src->cols / 2, CV_8UC3);
resize(small, src, INTER_AREA);           // 縮小用 INTER_AREA；也有 INTER_NEAREST、INTER_LINEAR
mat_write("small.jpg", small);
```

### 不使用堆積（微控制器）

在沒有 malloc 的環境，用 `mat_init` 把自己的緩衝區包成 `Mat`。只需要連結 `improc_core`，不需要檔案系統：

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

`cvt_color`、`threshold`、`mat_lut`、`convert_scale`、`absdiff`、`flip`、`crop`、`calc_hist`、`otsu_threshold_hist`、`nonzero_stats` 與畫圖函式（`draw_*`）不需要暫存記憶體。濾波（`median_blur` 只在 `ksize` 7 以上）、形態學、`resize`、`equalize_hist`、`otsu_threshold` 與 `fast_detect`（只在開啟非極大值抑制時）會向 `fast_malloc` 要一小塊暫存。沒有 malloc 時，要用 `improc_set_allocator` 提供自己的配置器，例如固定大小的記憶體池。各運算的暫存大小見 [docs/porting.md](docs/porting.md)。

### 動態偵測

`prev`、`curr` 是相機前後兩張 QVGA 灰階畫面（`CV_8UC1`，320×240）。這裡只看中央 160×120 的區域：

```c
crop(roi_prev, prev, 80, 60);                   // roi_*、diff、mask 都是 120×160 的 CV_8UC1
crop(roi_curr, curr, 80, 60);
absdiff(diff, roi_curr, roi_prev);
threshold(mask, diff, 30, 255, THRESH_BINARY);  // 亮度變化超過 30 的像素
struct NonzeroStats s;
nonzero_stats(&s, mask);
if (s.count > 200) {                            // 變動的像素夠多才算有動靜
    int cx = (int)((s.sum_x + s.count / 2) / s.count);  // 重心，四捨五入
    int cy = (int)((s.sum_y + s.count / 2) / s.count);
    // 物體範圍 (s.x_min, s.y_min)～(s.x_max, s.y_max)，座標相對於 ROI
}
```

完整的可執行範例是 `main.c`（建置後的 `demo`）。它對一張圖依序跑過所有運算，並各存一張 PNG：

```sh
./build/demo [input-image] [output-prefix]     # 預設使用內建測試圖，prefix 為 out_
./build/bench_gray                             # 量測 RGB 轉灰階的耗時；效能數字請用 Release 建置
```

## 功能

| 模組 | 函式 | 說明 |
|---|---|---|
| `color.h` | `cvt_color` | GRAY、RGB、BGR、RGBA、BGRA 互轉；灰階用 `(77R + 150G + 29B + 128) >> 8` |
| `color.h` | `cvt_color`（HSV、YUV） | RGB／BGR 與 HSV（H 0～179）互轉；RGB／BGR 與 YUV（JFIF 全範圍 BT.601）互轉 |
| `color.h` | `cvt_color`（相機格式） | RGB565（`CV_8UC2`，小端序與相機常用的大端序 `BE`）轉 BGR／RGB／GRAY；YUV422 YUYV 轉 BGR／RGB／GRAY（有限範圍 BT.601，與 OpenCV 逐 byte 相同，寬度須為偶數） |
| `pixel.h` | `threshold` | BINARY、BINARY_INV、TRUNC、TOZERO、TOZERO_INV |
| `pixel.h` | `mat_lut` | 256 項查表 |
| `pixel.h` | `absdiff` | 兩張影像逐通道相減取絕對值，用於畫面差分（動態偵測） |
| `pixel.h` | `convert_scale`、`convert_scale_abs` | `src * alpha_q8 / 256 + beta`，結果飽和到 0～255 |
| `filter.h` | `box_filter`、`gaussian_blur`、`sobel` | 邊界一律 BORDER_REFLECT_101；以逐列滾動的環形緩衝區運作，暫存記憶體只與 `ksize` 和列寬有關；Sobel 輸入灰階、輸出 `CV_16SC1` |
| `filter.h` | `median_blur` | 中值濾波，`ksize` 為 3～255 的奇數，邊界 BORDER_REPLICATE，結果與 `cv2.medianBlur` 逐位元組相同；`ksize` 3 與 5 用排序網路與選擇法，7 以上用 Huang 滑動直方圖（暫存 256 個 `uint16_t` 加 `ksize` 個列指標） |
| `morph.h` | `erode`、`dilate`、`morph_open`、`morph_close` | 矩形結構元素，`ksize` 為 1～255 的奇數；視窗超出影像的部分不計入 |
| `hist.h` | `calc_hist`、`equalize_hist` | 256 階直方圖與直方圖等化 |
| `hist.h` | `otsu_threshold`、`otsu_threshold_hist` | Otsu 門檻；判準以精確的整數與分數比較，沒有浮點捨入造成的平手 |
| `hist.h` | `nonzero_stats` | 二值影像的非零像素數、外接矩形與重心座標總和，用來判斷有沒有物體、在哪裡、多大 |
| `features2d.h` | `fast_detect` | FAST-9 角點偵測（灰階），可選擇非極大值抑制；角點寫進呼叫端提供的 `struct KeyPoint` 陣列，回傳找到的總數；結果與 `cv2.FastFeatureDetector` 的位置、順序、分數都相同 |
| `draw.h` | `draw_point`、`draw_line`、`draw_circle` | 在影像上畫點、線、圓，用來標示偵測結果；超出影像的部分直接裁掉；粗細 1 的線與圓、實心圓和 `cv2.line`／`cv2.circle`（`LINE_8`）逐位元組相同 |
| `draw.h` | `draw_text`、`draw_text_width` | 內建 5×8 點陣字型，只支援 ASCII 0x20～0x7E，可整數倍放大；其他位元組畫成方框，一個 UTF-8 字元只畫一個方框 |
| `geometry.h` | `flip` | 水平、垂直、雙向翻轉 |
| `geometry.h` | `crop` | 從大圖複製出一塊矩形區域（任何型別），區域大小由 dst 決定 |
| `geometry.h` | `resize` | `INTER_NEAREST`、`INTER_LINEAR`（半像素中心）與 `INTER_AREA`（整數面積權重，放大縮小皆可） |
| `convert_f32.h` | `convert_to` | `CV_8UCn` 與 `CV_32FCn` 互轉並套用 `alpha`、`beta`；需要 `IMPROC_ENABLE_FLOAT` |
| `mat.h` | `alloc_mat`、`mat_init` | `alloc_mat` 從堆積配置；`mat_init` 包裝呼叫端的緩衝區，不配置記憶體 |
| `mat_io.h` | `alloc_mat_from_file`、`mat_write` | 讀寫 PNG 與 JPG，依副檔名選格式（只在 `improc` 函式庫，不在 `improc_core`） |

## 測試與驗證

### 單元測試

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DIMPROC_WERROR=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

### Kodak 測試

`test_kodak_*` 用 24 張 Kodak 影像跑每個運算，把輸出的 FNV-1a 雜湊與 `tests/golden/kodak_*.txt` 比對，確認各平台逐 byte 相同。資料集不在版本庫內（`datasets/` 已被 git 忽略）。請自行從 <https://r0k.us/graphics/kodak/> 下載 `kodim01.png`～`kodim24.png`，放進 `datasets/kodak`。沒有資料集時，這些測試會顯示為 Skipped，其餘測試照常執行。要用其他位置時，設定環境變數 `IMPROC_KODAK_DIR`，或在 CMake 加 `-DIMPROC_KODAK_DIR=<目錄>`。

### 與 OpenCV 比對

比對腳本需要 Python 3、`opencv-python` 與 `numpy`（`pip install opencv-python numpy`）。先用各測試的 `--dump` 輸出影像，再讓腳本逐運算算出與 OpenCV 的最大差異：

```sh
for m in phase1 morph hist color2 camera area edge median features draw f32; do
  build/test_kodak_$m --dir datasets/kodak --manifest tests/golden/kodak_$m.txt --dump dump
done
python3 tools/compare_opencv.py --dump dump
```

每個運算有各自的容許誤差，超過就以非零狀態結束。結果與解讀見 [docs/kodak-results.md](docs/kodak-results.md)。

## 文件

| 文件 | 內容 |
|---|---|
| [docs/README.md](docs/README.md) | 模組關係、`struct Mat` 資料模型與各演算法流程的圖解 |
| [docs/porting.md](docs/porting.md) | 移植到邊緣裝置：編譯旗標、函式庫大小、各運算的暫存記憶體、沒有 malloc 時的用法 |
| [docs/kodak-results.md](docs/kodak-results.md) | 每個運算與 OpenCV 的差異和容許誤差 |

## 授權

MIT，見 [LICENSE](LICENSE)。`core/stb_image.h` 與 `core/stb_image_write.h` 是第三方檔案，沿用各自檔內的授權。它們只用於 `improc` 的圖檔讀寫，不在 `improc_core` 與 Arduino 函式庫內。

# 效能與最佳化

本文記錄各運算的速度、做過的 SIMD 與迴圈最佳化，以及怎麼確認最佳化前後的輸出逐位元組相同。

## 量測方法

`benchmarks/bench_ops.c` 對一張照片執行每個運算，每批約 20 ms，取 9 批的中位數，單位是每個來源像素的奈秒（ns/px）。它同時印出輸出的 FNV-1a 雜湊，最佳化前後的雜湊必須相同。

```sh
cmake -S . -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --target bench_ops
build-release/bench_ops datasets/kodak/kodim01.png            # 全部運算
build-release/bench_ops datasets/kodak/kodim01.png --only gauss5
```

下表的環境：Windows、LLVM-MinGW、Release、i7-12700，用 `start /affinity 1` 固定在一個效能核心（這顆 CPU 有大小核，不固定時同一個運算的時間會跳兩倍），影像 kodim01（768×512 RGB）。

## 結果

「之前」是最佳化前的同一份程式（已含 OpenCV 對齊的修正）。只列有變化的運算；其餘運算（色彩順序、YUV、查表、`absdiff`、`flip` 等）的時間不變，原本就在 0.03～2.5 ns/px。

| 運算 | 之前 ns/px | 之後 ns/px | 倍數 | 做法 |
|---|---|---|---|---|
| `median_blur` 5×5（RGB） | 529.7 | 4.45 | 119 | 內部像素用 SIMD 比較網路 |
| `median_blur` 3×3（RGB） | 31.3 | 0.87 | 36 | 同上 |
| `gaussian_blur` 3×3 | 9.39 | 0.88 | 10.7 | 約分權重、16 位元 SIMD、`pad_row` 用 `memcpy` |
| `gaussian_blur` 5×5 | 11.56 | 1.32 | 8.8 | 同上 |
| `gaussian_blur` 7×7 | 14.31 | 2.76 | 5.2 | 水平 16 位元、垂直 32 位元 SIMD |
| `cvt_color` HSV 轉 RGB | 91.8 | 10.4 | 8.8 | 64 位元主機改用硬體 float32 |
| `resize` 縮成一半（`INTER_AREA`） | 3.48 | 0.46 | 7.5 | 2×2 專用迴圈 |
| `sobel` | 0.62 | 0.14 | 4.5 | `pad_row` 用 `memcpy` |
| `dilate` 3×3 | 5.07 | 1.19 | 4.3 | 垂直與水平都用 SIMD min/max |
| `erode` 3×3 | 4.78 | 1.22 | 3.9 | 同上 |
| `morph_open` 5×5 | 13.57 | 2.77 | 4.9 | 同上 |
| `resize` 放大 1.5 倍（`INTER_AREA`） | 11.07 | 3.36 | 3.3 | 同 `INTER_LINEAR` |
| `resize` 放大 1.5 倍（`INTER_LINEAR`） | 10.93 | 3.58 | 3.1 | 水平和每列只算一次、垂直 SIMD |
| `box_filter` 5×5 | 6.86 | 2.58 | 2.7 | 除法改成乘法加位移 |

沒有變快的：`fast_detect`（16.5）、7×7 以上的 `median_blur`（Huang 直方圖，16.5）。FAST 試過用 SIMD 先篩掉整組 16 個非候選像素，但照片裡幾乎每組都有候選，只快 3%，所以沒有保留。

## 各項做法

`core/improc_simd.h` 是一層很薄的 128 位元向量介面：x86-64 用 SSE2，AArch64 與有 NEON 的 32 位元 ARM 用 NEON，其他目標（Cortex-M、ESP32、AVR）沒有向量路徑，執行原本的純量迴圈。向量迴圈只處理每列中能整組讀取的部分。中值與形態學的水平方向只算影像內部，邊界像素用純量程式，列尾的最後一組向量與前一組重疊，不需要另外的尾端迴圈；其他向量迴圈（高斯兩個方向、形態學垂直方向、雙線性垂直方向）把最後不足 16 個位元組的部分交給純量迴圈。

- 中值濾波：每個位元組是一個獨立的「像素-通道」，鄰居相距 `cn` 個位元組，所以 16 個位元組可以一起排序。比較網路由 `tools/gen_median_net.py` 產生：從 Batcher 奇偶合併排序出發，刪掉不影響中位數的比較器（9 個值剩 24 個比較器，25 個值剩 113 個），再依 0-1 原理窮舉全部 2^9 與 2^25 種 0/1 輸入驗證。網路以巨集展開，索引在編譯期就是常數。
- 高斯：原本的權重總和 256，兩個方向相乘後最大 65280×256，需要 32 位元。3×3 與 5×5 的權重是 64·(1,2,1) 與 16·(1,4,6,4,1)，約分後的二維和最多 4080 與 65280，原本的 `(acc + 32768) >> 16` 恰好等於 `(S + 8) >> 4` 與 `(S + 128) >> 8`。這是整數恆等式，所以輸出不變，兩個方向都能用 16 位元向量；純量版也改用約分後的權重，AVR 上同樣受益。7×7 約分後仍需要 32 位元，垂直方向用 32 位元向量累加。
- box：每個輸出除以面積改成乘以 `m = floor(2^s / 面積) + 1` 再右移 `s = 8 + 2L`（`2^L ≥ 面積`）。被除數小於 `256 × 面積 ≤ 2^(8+L)`，誤差項小於 `1/面積`，所以結果等於整數除法。ksize 1～255 的所有面積與所有可能的被除數都窮舉驗證過。
- `pad_row`（box、高斯、Sobel 共用）：原本每個像素都做一次 reflect 101 計算，現在中間部分用 `memcpy`，只有兩端各 `r` 個像素做反射。
- 雙線性縮放：原本每個輸出像素都重算兩列的水平和。現在每列來源的水平和只算一次，存成 `h >> 4` 的 `int16_t`（垂直步驟只讀這個值），兩個緩衝區依來源列號快取；垂直步驟用 `mulhi`，也就是 OpenCV 自己的演算法。暫存因此多了 `4 × dst_cols × cn` 位元組。
- HSV 轉 RGB：可攜版本用整數模擬 OpenCV 的 float32 運算，每像素十幾次 64 位元捨入。在 x86-64 與 AArch64（`FLT_EVAL_METHOD == 0`）上改用硬體 float32，`1 - a·b` 的單次捨入用 double 與 TwoSum 求出，不需要 FMA 指令。兩條路徑都與 `tests/test_color.c` 的參考實作在全部 2^24 種輸入上比對。

## 輸出不變的驗證

- Kodak golden：最佳化沒有更動任何 `tests/golden/kodak_*.txt`，十個模組的雜湊全部通過。
- `bench_ops` 的雜湊：每個運算在 kodim01 上的輸出雜湊與最佳化前相同。唯一的例外是 `linear_nonint`，它的基準線量在雙線性改用 float32 座標之前；這個尺寸就是 Kodak 的 `nonint_linear`，由 golden 確認輸出不變。
- 單元測試：寬度 0～65、1／3／4 通道的亂數影像（`fill_xorshift`），中值、高斯、形態學與 naive 參考比對；HSV 轉 RGB 窮舉 2^24 種輸入。刻意弄壞比較網路或形態學視窗時，這些測試都會失敗。
- 三種組態：WSL GCC（SSE2）、`IMPROC_NO_SIMD`（純量與整數 HSV）、qemu-aarch64（NEON 與 AArch64 float HSV）的 ctest 全部通過；ASan 與 UBSan 下也通過（`test_io` 的 UBSan 報錯在第三方 `stb_image_write.h`，與本專案無關）。

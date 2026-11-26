# Kodak 測試與 OpenCV 比對結果

本文記錄在 Kodak 24 張影像上，每個運算與 OpenCV 的最大差異。資料集來源與下載方式見 [README.md](../README.md)。

## 做法

1. 每個模組各有一支 `test_kodak_<module>`。它對 24 張影像各跑一遍運算，計算輸出的 FNV-1a 雜湊，與 `tests/golden/kodak_<module>.txt` 比對。同一份雜湊清單在 Windows（LLVM-MinGW）、WSL（GCC x86-64）與 `qemu-aarch64` 下都通過，所以各平台的輸出逐 byte 相同。
2. 加上 `--dump <目錄>` 時，測試把每個輸出存成 `<影像>_<運算>.png`。十一個模組倒進同一個目錄。
3. `tools/compare_opencv.py --dump <目錄>` 讀入這些 PNG。每個運算用我們自己輸出的輸入餵給 OpenCV（例如 `cv2.Sobel` 的輸入是我們輸出的灰階圖），所以某一步的差異不會傳到下一步。最大差異超過容許誤差就以非零狀態結束。

```sh
for m in phase1 morph hist color2 camera area edge median features draw f32; do
  build/test_kodak_$m --dir datasets/kodak --manifest tests/golden/kodak_$m.txt --dump dump
done
python3 tools/compare_opencv.py --dump dump
```

`f32`（`CV_8U` 轉 `CV_32F` 再轉回的恆等性）沒有對應的 OpenCV 運算，只驗證雜湊，不在比對表內。`draw_text` 也一樣：`cv2.putText` 用向量字型，與本專案的點陣字型無從比較，所以只驗證雜湊。

## 結果

OpenCV 4.10.0，24 張影像，53 個運算，全部在容許誤差內。「差異像素」是所有影像的樣本中與 OpenCV 不同的比例。
OpenCV 4.10.0，24 張影像，51 個運算，全部在容許誤差內。「差異像素」是所有影像的樣本中與 OpenCV 不同的比例。

| 運算 | 模組 | 最大差異 | 容許 | 差異像素 % | 對照的 OpenCV 呼叫 |
|---|---|---|---|---|---|
| `gray` | phase1 | 0 | 0 | 0 | `COLOR_RGB2GRAY` |
| `bgr` | phase1 | 0 | 0 | 0 | `COLOR_RGB2BGR` |
| `rgba` | phase1 | 0 | 0 | 0 | `COLOR_RGB2RGBA` |
| `thresh128` | phase1 | 0 | 0 | 0 | `THRESH_BINARY` |
| `invert` | phase1 | 0 | 0 | 0 | `LUT` |
| `bright_384_10` | phase1 | 0 | 0 | 0 | `convertScaleAbs(1.5, 10)` |
| `box5` | phase1 | 0 | 0 | 0 | `cv2.blur` 5x5 |
| `gauss3`、`gauss5`、`gauss7` | phase1 | 0 | 0 | 0 | `GaussianBlur` |
| `sobel_x_abs`、`sobel_y_abs` | phase1 | 0 | 0 | 0 | `Sobel`（`CV_16S`，`ksize` 3） |
| `flip_h`、`flip_v`、`flip_both` | phase1 | 0 | 0 | 0 | `flip` |
| `half_linear` | phase1 | 0 | 0 | 0 | `INTER_LINEAR`（剛好一半時 OpenCV 改走 `INTER_AREA`） |
| `nonint_linear` | phase1 | 0 | 0 | 0 | `INTER_LINEAR` |
| `double_nearest` | phase1 | 0 | 0 | 0 | `INTER_NEAREST_EXACT` |
| `erode3` | morph | 0 | 0 | 0 | `cv2.erode` 3x3 |
| `dilate3` | morph | 0 | 0 | 0 | `cv2.dilate` 3x3 |
| `open5` | morph | 0 | 0 | 0 | `morphologyEx` OPEN 5x5 |
| `close5` | morph | 0 | 0 | 0 | `morphologyEx` CLOSE 5x5 |
| `gray_erode7` | morph | 0 | 0 | 0 | `cv2.erode` 7x7 |
| `equalize` | hist | 0 | 0 | 0 | `equalizeHist` |
| `otsu_binary` | hist | 0 | 0 | 0 | `THRESH_OTSU`（同時比對直方圖與 Otsu 門檻值） |
| `rgb2hsv` | color2 | 0 | 0 | 0 | `COLOR_RGB2HSV` |
| `hsv2rgb_roundtrip` | color2 | 0 | 0 | 0 | `COLOR_HSV2RGB` |
| `rgb2yuv` | color2 | 0 | 0 | 0 | `COLOR_RGB2YCrCb`（色度通道對調） |
| `yuv2rgb_roundtrip` | color2 | 0 | 0 | 0 | `COLOR_YCrCb2RGB`（色度通道對調） |
| `bgr565_rgb` | camera | 0 | 0 | 0 | `COLOR_BGR5652RGB` |
| `bgr565be_gray` | camera | 0 | 0 | 0 | `COLOR_BGR5652GRAY` |
| `yuyv_rgb` | camera | 0 | 0 | 0 | `COLOR_YUV2RGB_YUYV` |
| `yuyv_gray` | camera | 0 | 0 | 0 | `COLOR_YUV2GRAY_YUYV` |
| `area_half` | area | 0 | 0 | 0 | `INTER_AREA` |
| `area_third` | area | 1 | 1 | 0.0156 | `INTER_AREA` |
| `area_quarter` | area | 0 | 0 | 0 | `INTER_AREA` |
| `area_nonint` | area | 1 | 1 | 0.0004 | `INTER_AREA` |
| `area_up_1_5` | area | 0 | 0 | 0 | `INTER_AREA` |
| `median3` | median | 0 | 0 | 0 | `cv2.medianBlur` 3 |
| `median5` | median | 0 | 0 | 0 | `cv2.medianBlur` 5 |
| `gray_median7` | median | 0 | 0 | 0 | `cv2.medianBlur` 7（我們的灰階） |
| `gray_median15` | median | 0 | 0 | 0 | `cv2.medianBlur` 15（我們的灰階） |
| `fast` | features | 0 | 0 | 0 | `FastFeatureDetector` 9_16，門檻 20，非極大值抑制（角點位置與分數） |
| `fast_raw` | features | 0 | 0 | 0 | `FastFeatureDetector` 9_16，門檻 20，不抑制（角點位置） |
| `half` | sift | 0 | 0 | 0 | `INTER_AREA` 縮小一半（SIFT 的輸入） |
| `sift_desc` | sift | 1 | 1 | 0.0964 | `SIFT.compute`，在我們的關鍵點上算描述子；同時檢查 `sift_kps` |
| `rotscale` | sift | 0 | 0 | 0 | 我們自己的旋轉縮放；同時檢查 `sift_rotscale` 的配對正確率 |
| `edge_gray` | edge | 1 | 1 | 3.6653 | `COLOR_RGB2GRAY` |
| `crop_center` | edge | 0 | 0 | 0 | numpy 切片（中央一半） |
| `absdiff_shift` | edge | 0 | 0 | 0 | `cv2.absdiff`（相距 6 像素的兩個畫面） |
| `motion_mask` | edge | 0 | 0 | 0 | `absdiff` 後 `THRESH_BINARY` 40 |
| `motion_box` | edge | 0 | 0 | 0 | numpy 外框與重心（同時比對 `motion_stats`、`still_stats`） |
| `corner_box` | edge | 0 | 0 | 0 | 角落 12×12，十字被邊界裁切（同時比對 `corner_stats`） |
| `draw_lines` | draw | 0 | 0 | 0 | `cv2.line`，`LINE_8`，粗細 1（含超出影像與座標 ±2^24 的線） |
| `draw_circles` | draw | 0 | 0 | 0 | `cv2.circle`，`LINE_8`，粗細 1 與實心（含半徑 2^20 的圓） |
| `draw_thick` | draw | 0 | 0 | 0 | 本專案的粗線與圓環規則（`cv2.line` 加圓形筆刷 `cv2.dilate`、numpy 圓環），不是 OpenCV 的粗線 |

## 解讀

- 逐 byte 相同（最大差異 0）的運算：色彩順序轉換、查表、濾波（box、gauss、Sobel）、翻轉、裁切、絕對差、整數倍的 nearest、中值濾波、FAST 角點偵測、粗細 1 的畫線與畫圓、實心圓、形態學、直方圖等化、Otsu、RGB 轉 HSV、半倍縮小的 `INTER_AREA`。`equalize`、`half_linear`、`area_up_1_5` 的容許誤差是 1，實測也是 0。
- 差 1 個階的運算都是定點捨入與 OpenCV 的捨入方式不同：灰階、`convertScaleAbs`、非整數倍的雙線性縮放、HSV 轉回 RGB、YUV、三分之一與四分之一的 `INTER_AREA`。差異都在容許範圍內，沒有放寬任何容許誤差。
- 相機格式（camera 模組）的輸入不是真的相機畫面：測試把 Kodak 的 RGB 打包成 RGB565 或 YUYV 位元組，再交給解碼函式。`tools/opencv_ref/camera.py` 用同樣的方式打包，再交給 OpenCV。RGB565 與 YUYV 轉 RGB 和 OpenCV 逐 byte 相同；RGB565 轉灰階沿用本專案的灰階公式，所以和 OpenCV 差 1。
- FAST 的結果是角點清單，不是影像。features 模組把角點畫成同尺寸的 `CV_8UC1` 圖：有非極大值抑制時，角點像素存分數；沒有時存 255；其餘像素為 0。`tools/opencv_ref/features.py` 用 OpenCV 的角點畫出同樣的圖，所以最大差異 0 表示角點的位置與分數都相同。
- SIFT 用浮點運算，但 exp、atan2、sin、cos 都是自己寫的多項式，並且關閉 FMA，所以各平台的雜湊仍然相同。這些函式和 libm 不同，所以結果和 `cv2.SIFT` 不是逐位元組相同：
  - 關鍵點：在縮小一半的 24 張影像上，`cv2.SIFT` 的關鍵點有 99.7%～100% 能在 1 像素內找到我們的關鍵點（尺度比 0.8～1.25、角度差 10 度以內），反過來也是 99.9%～100%。數量最多差 1 個。
  - 描述子：把我們的關鍵點交給 `cv2.SIFT.compute`，兩邊的描述子最多差 1，只有 0.0964% 的 byte 不同。
  - 配對：把影像旋轉 30 度、縮小為 0.75 倍，用比值檢定（0.8）配對，配對點落在真正位置 3 像素內算正確。我們 5137/5528（92.9%），`cv2.SIFT` 5133/5523（92.9%）。任何一張影像的正確率比 OpenCV 低超過 5 個百分點時，參考函式判定失敗；`python3 tools/opencv_ref/sift.py --dump <目錄>` 會印出每張影像的數字。
- 每個運算的容許誤差寫在 `tools/opencv_ref/*.py`。

## 對齊 OpenCV 的演算法

先前有七個運算與 OpenCV 差 1 個階，差異像素介於 3% 與 31% 之間，原因是公式與 OpenCV 不同。我們用窮舉輸入找出 OpenCV 實際的整數演算法，再照著改寫：

| 運算 | 原本的做法 | 改成（OpenCV 4.10 的做法） |
|---|---|---|
| 灰階、RGB565 灰階 | Q8 權重 77/150/29 | Q15 權重 9798/19235/3735 |
| RGB 與 YUV 互轉 | JFIF 全範圍、Q8 | `YCrCb` 的 Q14 常數；Cr、Cb 由捨入後的 Y 算出 |
| `convert_scale`、`convert_scale_abs` | 四捨五入（0.5 進位） | 四捨六入五成雙 |
| HSV 轉 RGB | 精確值四捨五入 | float32 計算，`1 - s·h` 為融合乘加（FMA），乘 255 後無條件捨去；本專案用整數模擬這些 float32 運算 |
| `INTER_LINEAR` | Q8 權重，兩次加權後一次捨入 | 來源位置先捨入成 float32，再取整數與小數部分；Q11 權重由小數各自捨入；x 夾邊界、y 不夾；垂直方向兩個乘積各自截斷後再捨入。float32 的捨入用整數模擬 |
| `INTER_AREA` 縮小 | 平均值四捨五入 | 四捨六入五成雙；只有剛好 2×2 時進位 |
| `INTER_AREA` 放大 | 面積平均 | 雙線性核心加上 OpenCV 的 area 座標公式 |

HSV 轉 RGB 有一個例外，影響寬度不是 32 倍數的影像。OpenCV 每列以 32 個像素為一組走向量迴圈，最後 `寬度 % 32` 個像素改走純量迴圈。純量迴圈會四捨五入，結果和向量迴圈不同，這些像素約三分之一差 1。本專案每一欄都照向量迴圈的演算法，不模擬這個與位置有關的行為。Kodak 的寬度 768 與 512 都是 32 的倍數，所以不受影響。寬度 500 時，每列最後 20 個像素（4%）落在純量迴圈，整張圖約 1.3% 的樣本和 OpenCV 不同，超過 1%。32 是在 x86-64（AVX2）上量到的值；`tools/sweep_opencv.py` 會在執行時偵測這個寬度。

代價有三項：

- 灰階的乘加需要 32 位元整數，在 16 位元 `int` 的 AVR 上比較慢。
- HSV 轉 RGB 改成無條件捨去後，RGB→HSV→RGB 來回的最大誤差從 5 變成 6。
- HSV 轉 RGB 的整數 float32 模擬比原本的公式慢很多：kodim01 上從 2.7 ns/px 變成 91 ns/px（Release，LLVM-MinGW，i7-12700 固定單一核心）。

## 廣泛測試（不只 Kodak）

Kodak 只代表一種色彩分布。`tools/sweep_opencv.py` 用窮舉與隨機輸入比對，經 `improc_probe`（`tools/improc_probe.c`）執行本專案的運算：

```sh
cmake --build build --target improc_probe
python3 tools/sweep_opencv.py --probe build/improc_probe
```

色彩轉換用全部輸入值（2^24 種 RGB、2^16 種 565 字組、256^3 種 HSV）。縮放用固定種子的隨機影像與隨機尺寸，含 1、3、4 通道。任一列超過上限就以非零狀態結束。下表是 OpenCV 4.10.0（x86-64）的結果。

| 運算 | 輸入 | 修正前差異 % | 修正後差異 % | 上限 % |
|---|---|---|---|---|
| `RGB2GRAY` 等四種灰階 | 全部 2^24 種 RGB | 13.2991 | 0 | 0 |
| `BGR5652GRAY`、`BGR565BE2GRAY` | 全部 2^16 種字組 | 13.5300 | 0 | 0 |
| `RGB2YUV`、`BGR2YUV` | 全部 2^24 種 RGB | 15.7250 | 0 | 0 |
| `YUV2RGB`、`YUV2BGR` | 全部 2^24 種 YUV | 2.6557 | 0 | 0 |
| `convert_scale` | 256 種輸入 × 1782 組參數 | 0.3393 | 0 | 0 |
| `convert_scale_abs` | 2^16 種 `CV_16S` × 48 組參數 | 0.1163 | 0 | 0 |
| `HSV2RGB`、`HSV2BGR` | 全部 256^3 種 HSV，寬 4096 | 33.3907 | 0 | 0 |
| `HSV2RGB` | 全部 2^24 種 RGB 轉 HSV 後轉回 | 34.0337 | 0 | 0 |
| `HSV2RGB` | 寬 1～100，向量迴圈的欄 | 33.2454 | 0 | 0 |
| `HSV2RGB` | 寬 1～100，純量列尾的欄 | 0.0226 | 33.3810 | 只回報 |
| `INTER_LINEAR` | 80 組隨機尺寸 1～300 | 16.2021 | 0 | 1 |
| `INTER_LINEAR` | Kodak 尺寸 | 16.2141 | 0 | 1 |
| `INTER_LINEAR` | 整數倍放大 | 11.9432 | 0 | 1 |
| `INTER_LINEAR` | 大尺寸（到 4000×3000） | 15.9699 | 0 | 1 |
| `INTER_AREA` | 整數倍縮小（1～8 倍） | 8.1145 | 0 | 1 |
| `INTER_AREA` | 整數倍縮小，面積 28～169 | 0.3546 | 0.1013 | 1 |
| `INTER_AREA` | 非整數倍縮小 | 0.0378 | 0.0044 | 1 |
| `INTER_AREA` | Kodak 尺寸縮小 | 0.4538 | 0.0046 | 1 |
| `INTER_AREA` | 放大約 1.5 倍 | 7.7293 | 0.1031 | 1 |
| `INTER_AREA` | 隨機尺寸放大 | 7.0175 | 0.2258 | 1 |
| `INTER_AREA` | 放大，gcd 大的尺寸 | 2.1333 | 0 | 1 |
| `INTER_AREA` | 一軸放大、一軸縮小 | 34.9379 | 0.2597 | 1 |
| `INTER_AREA` | 剛好放大 1.5 倍 | 0 | 0 | 1 |

`INTER_LINEAR` 在寬高 4096 以下的所有測試尺寸都與 OpenCV 相同；更大的尺寸只在精確位置剛好落在 float32 中點時可能差 1（審查時量了寬 8192～91629 的幾組，差異樣本不到 0.08%）。OpenCV 先用 double 算出來源位置，再轉成 float32；位置越大，float32 的間隔越粗，所以這一步在大圖上一定要照做（只用精確有理數時，4000×3000 縮成 3333×2222 會差 1.86%）。`tests/test_geometry.c` 照 OpenCV 原始碼用 double 與 float 重算約 2000 組尺寸的每個索引與權重，與本專案的整數版比對。

`INTER_AREA` 剩下的差異有兩種。整數倍縮小（剛好 2×2 以外）時，OpenCV 計算 `float(和) × float(1/面積)` 再捨入，面積大時偶爾和精確平均差 1。放大時，精確的來源位置剛好是整數的欄，OpenCV 的 double 運算可能算成小一點的值，向下取整後取到前一個像素，那一欄就整欄不同（最大差異可到 249）。本專案在這兩處都用精確的整數。

## ORB 的配對品質

本專案的 ORB 不和 `cv2.ORB` 逐位元組比對，原因有兩個。第一，256 組取樣點由專案自己的種子產生器產生（`tools/gen_orb_pattern.py`），不是 OpenCV 學習得來的表。第二，方向量化成 32 個角度，只用整數計算，不用 `atan2`。所以這裡比較的是配對品質。

做法：兩邊都對我們的灰階圖和它旋轉 30° 的版本（`rot30`，整數雙線性內插）各取 500 個關鍵點，只用一個尺度，FAST 門檻 20，再用 Hamming 距離做交叉檢查配對。配對點落在真實旋轉位置 3 像素以內，就算正確。`tools/opencv_ref/features.py` 在 `compare_opencv.py` 中逐張檢查，差距不得超過 10 個百分點；直接執行它，會印出逐張表格與總計，總計差距不得超過 3 個百分點。

| | 正確／配對 | 正確比例 | 單張範圍 |
|---|---|---|---|
| 本專案 `orb_detect` + `orb_compute` | 7552／7886 | 95.8% | 85.7%～99.5% |
| `cv2.ORB_create(500, nlevels=1)` | 7889／8221 | 96.0% | 86.1%～99.5% |

正確比例相近，24 張中有 10 張高於 OpenCV。配對數少約 4%。差距最大的是 kodim15（90.4%，OpenCV 98.4%）。兩邊都只用一個尺度，所以都不處理縮放。

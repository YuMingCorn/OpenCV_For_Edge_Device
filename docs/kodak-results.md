# Kodak 測試與 OpenCV 比對結果

本文記錄在 Kodak 24 張影像上，每個運算與 OpenCV 的最大差異。資料集來源與下載方式見 [README.md](../README.md)。

## 做法

1. 每個模組各有一支 `test_kodak_<module>`。它對 24 張影像各跑一遍運算，計算輸出的 FNV-1a 雜湊，與 `tests/golden/kodak_<module>.txt` 比對。同一份雜湊清單在 Windows（LLVM-MinGW）、WSL（GCC x86-64）與 `qemu-aarch64` 下都通過，所以各平台的輸出逐 byte 相同。
2. 加上 `--dump <目錄>` 時，測試把每個輸出存成 `<影像>_<運算>.png`。十一個模組倒進同一個目錄。
3. `tools/compare_opencv.py --dump <目錄>` 讀入這些 PNG。每個運算用我們自己輸出的輸入餵給 OpenCV（例如 `cv2.Sobel` 的輸入是我們輸出的灰階圖），所以某一步的差異不會傳到下一步。最大差異超過容許誤差就以非零狀態結束。

```sh
for m in phase1 morph hist color2 camera area edge median features f32 sift; do
  build/test_kodak_$m --dir datasets/kodak --manifest tests/golden/kodak_$m.txt --dump dump
done
python3 tools/compare_opencv.py --dump dump
```

`f32`（`CV_8U` 轉 `CV_32F` 再轉回的恆等性）沒有對應的 OpenCV 運算，只驗證雜湊，不在比對表內。sift 模組的 `sift_kps`（關鍵點清單）與 `sift_rotscale`（配對統計）不是影像，寫進 `values_sift.txt`，由 `sift_desc` 與 `rotscale` 兩列的參考函式一起檢查（見下方「解讀」）。

## 結果

OpenCV 4.10.0，24 張影像，53 個運算，全部在容許誤差內。「差異像素」是所有影像的樣本中與 OpenCV 不同的比例。
OpenCV 4.10.0，24 張影像，51 個運算，全部在容許誤差內。「差異像素」是所有影像的樣本中與 OpenCV 不同的比例。

| 運算 | 模組 | 最大差異 | 容許 | 差異像素 % | 對照的 OpenCV 呼叫 |
|---|---|---|---|---|---|
| `gray` | phase1 | 1 | 1 | 3.6653 | `COLOR_RGB2GRAY` |
| `bgr` | phase1 | 0 | 0 | 0 | `COLOR_RGB2BGR` |
| `rgba` | phase1 | 0 | 0 | 0 | `COLOR_RGB2RGBA` |
| `thresh128` | phase1 | 0 | 0 | 0 | `THRESH_BINARY` |
| `invert` | phase1 | 0 | 0 | 0 | `LUT` |
| `bright_384_10` | phase1 | 1 | 1 | 21.8100 | `convertScaleAbs(1.5, 10)` |
| `box5` | phase1 | 0 | 0 | 0 | `cv2.blur` 5x5 |
| `gauss3`、`gauss5`、`gauss7` | phase1 | 0 | 0 | 0 | `GaussianBlur` |
| `sobel_x_abs`、`sobel_y_abs` | phase1 | 0 | 0 | 0 | `Sobel`（`CV_16S`，`ksize` 3） |
| `flip_h`、`flip_v`、`flip_both` | phase1 | 0 | 0 | 0 | `flip` |
| `half_linear` | phase1 | 0 | 1 | 0 | `INTER_LINEAR` |
| `nonint_linear` | phase1 | 1 | 1 | 12.2109 | `INTER_LINEAR` |
| `double_nearest` | phase1 | 0 | 0 | 0 | `INTER_NEAREST_EXACT` |
| `erode3` | morph | 0 | 0 | 0 | `cv2.erode` 3x3 |
| `dilate3` | morph | 0 | 0 | 0 | `cv2.dilate` 3x3 |
| `open5` | morph | 0 | 0 | 0 | `morphologyEx` OPEN 5x5 |
| `close5` | morph | 0 | 0 | 0 | `morphologyEx` CLOSE 5x5 |
| `gray_erode7` | morph | 0 | 0 | 0 | `cv2.erode` 7x7 |
| `equalize` | hist | 0 | 1 | 0 | `equalizeHist` |
| `otsu_binary` | hist | 0 | 0 | 0 | `THRESH_OTSU`（同時比對直方圖與 Otsu 門檻值） |
| `rgb2hsv` | color2 | 0 | 0 | 0 | `COLOR_RGB2HSV` |
| `hsv2rgb_roundtrip` | color2 | 1 | 1 | 30.9841 | `COLOR_HSV2RGB` |
| `rgb2yuv` | color2 | 1 | 1 | 10.6839 | `COLOR_RGB2YCrCb`（色度通道對調） |
| `yuv2rgb_roundtrip` | color2 | 1 | 1 | 0.3657 | `COLOR_YCrCb2RGB`（色度通道對調） |
| `bgr565_rgb` | camera | 0 | 0 | 0 | `COLOR_BGR5652RGB` |
| `bgr565be_gray` | camera | 1 | 1 | 5.7574 | `COLOR_BGR5652GRAY`（我們用 77/150/29 的灰階公式） |
| `yuyv_rgb` | camera | 0 | 0 | 0 | `COLOR_YUV2RGB_YUYV` |
| `yuyv_gray` | camera | 0 | 0 | 0 | `COLOR_YUV2GRAY_YUYV` |
| `area_half` | area | 0 | 0 | 0 | `INTER_AREA` |
| `area_third` | area | 1 | 1 | 0.0469 | `INTER_AREA` |
| `area_quarter` | area | 1 | 1 | 3.0672 | `INTER_AREA` |
| `area_nonint` | area | 1 | 1 | 0.0009 | `INTER_AREA` |
| `area_up_1_5` | area | 0 | 1 | 0 | `INTER_AREA` |
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

## 解讀

- 逐 byte 相同（最大差異 0）的運算：色彩順序轉換、查表、濾波（box、gauss、Sobel）、翻轉、裁切、絕對差、整數倍的 nearest、中值濾波、FAST 角點偵測、形態學、直方圖等化、Otsu、RGB 轉 HSV、半倍縮小的 `INTER_AREA`。`equalize`、`half_linear`、`area_up_1_5` 的容許誤差是 1，實測也是 0。
- 差 1 個階的運算都是定點捨入與 OpenCV 的捨入方式不同：灰階、`convertScaleAbs`、非整數倍的雙線性縮放、HSV 轉回 RGB、YUV、三分之一與四分之一的 `INTER_AREA`。差異都在容許範圍內，沒有放寬任何容許誤差。
- 相機格式（camera 模組）的輸入不是真的相機畫面：測試把 Kodak 的 RGB 打包成 RGB565 或 YUYV 位元組，再交給解碼函式。`tools/opencv_ref/camera.py` 用同樣的方式打包，再交給 OpenCV。RGB565 與 YUYV 轉 RGB 和 OpenCV 逐 byte 相同；RGB565 轉灰階沿用本專案的灰階公式，所以和 OpenCV 差 1。
- FAST 的結果是角點清單，不是影像。features 模組把角點畫成同尺寸的 `CV_8UC1` 圖：有非極大值抑制時，角點像素存分數；沒有時存 255；其餘像素為 0。`tools/opencv_ref/features.py` 用 OpenCV 的角點畫出同樣的圖，所以最大差異 0 表示角點的位置與分數都相同。
- SIFT 用浮點運算，但 exp、atan2、sin、cos 都是自己寫的多項式，並且關閉 FMA，所以各平台的雜湊仍然相同。這些函式和 libm 不同，所以結果和 `cv2.SIFT` 不是逐位元組相同：
  - 關鍵點：在縮小一半的 24 張影像上，`cv2.SIFT` 的關鍵點有 99.7%～100% 能在 1 像素內找到我們的關鍵點（尺度比 0.8～1.25、角度差 10 度以內），反過來也是 99.9%～100%。數量最多差 1 個。
  - 描述子：把我們的關鍵點交給 `cv2.SIFT.compute`，兩邊的描述子最多差 1，只有 0.0964% 的 byte 不同。
  - 配對：把影像旋轉 30 度、縮小為 0.75 倍，用比值檢定（0.8）配對，配對點落在真正位置 3 像素內算正確。我們 5137/5528（92.9%），`cv2.SIFT` 5133/5523（92.9%）。任何一張影像的正確率比 OpenCV 低超過 5 個百分點時，參考函式判定失敗；`python3 tools/opencv_ref/sift.py --dump <目錄>` 會印出每張影像的數字。
- 每個運算的容許誤差寫在 `tools/opencv_ref/*.py`。

## ORB 的配對品質

本專案的 ORB 不和 `cv2.ORB` 逐位元組比對，原因有兩個。第一，256 組取樣點由專案自己的種子產生器產生（`tools/gen_orb_pattern.py`），不是 OpenCV 學習得來的表。第二，方向量化成 32 個角度，只用整數計算，不用 `atan2`。所以這裡比較的是配對品質。

做法：兩邊都對我們的灰階圖和它旋轉 30° 的版本（`rot30`，整數雙線性內插）各取 500 個關鍵點，只用一個尺度，FAST 門檻 20，再用 Hamming 距離做交叉檢查配對。配對點落在真實旋轉位置 3 像素以內，就算正確。`tools/opencv_ref/features.py` 在 `compare_opencv.py` 中逐張檢查，差距不得超過 10 個百分點；直接執行它，會印出逐張表格與總計，總計差距不得超過 3 個百分點。

| | 正確／配對 | 正確比例 | 單張範圍 |
|---|---|---|---|
| 本專案 `orb_detect` + `orb_compute` | 7553／7893 | 95.7% | 85.4%～99.5% |
| `cv2.ORB_create(500, nlevels=1)` | 7905／8246 | 95.9% | 86.1%～99.4% |

正確比例相近，24 張中有 11 張高於 OpenCV。配對數少約 4%。差距最大的是 kodim15（90.5%，OpenCV 98.1%）。兩邊都只用一個尺度，所以都不處理縮放。

# mini-album fixtures — 来源与许可证

顶接缝 e2e 测试用迷你相册（工单 13/14 的测试地基）。**非用户照片**，均为公开可再分发来源；
每张 ≤300KB。制备：qa（2026-10-21）。下载纪律：curl --max-time 45~60，逐源换源。

| 文件 | 内容 | 尺寸 | 字节 | 来源 URL | 许可证 |
|---|---|---|---|---|---|
| `portrait-01-frontal-clear.jpg` | 单人清晰正脸（ Barack Obama 官方总统肖像） | 820×1024 | 176,561 | https://storage.googleapis.com/mediapipe-assets/portrait.jpg | **公有领域**（美国联邦政府作品，白宫官方肖像）；经 Google MediaPipe 项目测试资产桶分发 |
| `portrait-02-frontal-clear.jpg` | 单人清晰正脸（商务人像，白底） | 958×1358 | 234,826 | https://storage.googleapis.com/mediapipe-assets/business-person.png | **MediaPipe 项目分发测试数据**（MediaPipe 仓库 Apache-2.0）；qa 以 sips 转 JPEG（quality 82）压至 ≤300KB |
| `group-01-three-frontal.jpg` | 3 人合影、正脸清晰（Apollo 11 机组官方合影：Armstrong / Collins / Aldrin） | 1280×1027 | 172,340 | https://images-assets.nasa.gov/image/S69-31740/S69-31740~medium.jpg | **公有领域**（NASA 图片，美国联邦政府作品，NASA ID S69-31740） |
| `eyes-closed-01-negative.jpg` | 单人正脸、双眼闭合（EAR 负例） | 1000×667 | 69,521 | https://www.pexels.com/photo/8955459/ （直链 images.pexels.com/photos/8955459/） | **Pexels License**（免费使用与再分发，无需署名） |
| `waste-blur-01.jpg` | 模糊废片（清晰度维度反例） | 820×1024 | 54,687 | 派生：`portrait-01` 经 ffmpeg `gblur=sigma=7` | 派生自公有领域作品（PD derivative） |
| `lowres-01.jpg` | 低分辨率小图（超分/质检维度） | 113×160 | 6,924 | 派生：`portrait-02` 经 sips `-Z 160` | 派生自 MediaPipe 项目分发测试数据 |
| `smile-teeth-01-frontal.jpg` | 单人正脸、露齿微笑、双眼睁开（whiten_teeth 功能正例） | 1000×1500 | 248,712 | https://www.pexels.com/photo/1181686/ （直链 images.pexels.com/photos/1181686/） | **Pexels License**（免费使用与再分发，无需署名） |
| `smile-teeth-02-pd-crop.jpg` | 露齿微笑正脸裁切（与 portrait-01 同人，跨图同身份+牙区素材） | 420×420 | 50,105 | 派生：`portrait-01` 经 sips 裁切（cropOffset 40,240 / 420×420） | 派生自公有领域作品（PD derivative） |

## 用途映射

- `portrait-01` / `portrait-02`：单人正脸——人脸检测/landmark/EAR 正例、磨皮/增强/调色的单对象输入。
- `group-01`：2~3 人合影——人物聚类（多脸同框）、按人寻址（cluster_id）的输入。
- `eyes-closed-01-negative`：EAR 闭眼负例（与两张 portrait 正例配对）。
- `smile-teeth-01/02`：露齿微笑——whiten_teeth 功能正例（牙 mask 非空、牙区 V 通道上升、唇区不动）；
  02 与 portrait-01 同人，兼作跨图同身份聚类素材。
- `waste-blur-01` / `lowres-01`：选片质检淘汰维度（清晰度/分辨率）反例；lowres 兼作超分输入。
- 合成辅助件（渐变图/40MP 大图/截断坏图/同人脸裁切）在 var/qa/synth/（gitignored），
  来源见 var/qa/synth/SOURCES.md。

## 制备期拒用记录（版权把关）

- insightface 仓库 `python-package/insightface/data/images/t1.jpg` = 《老友记》华纳宣传照 → 拒（无再分发许可）。
- dlib `examples/faces/bald_guys.jpg` = 名人新闻照拼贴 → 拒（各原图许可不清）。
- Unsplash `photo-1529156069898-49953e39b3ac`（8 人背身，无人脸）、`photo-1522071820081-009f0129c71c`（多人背/侧身）→ 弃（不满足正脸检出需求，非许可问题）。
- Pexels `1858175`（露齿大笑但双眼闭合+手部遮挡）→ 弃（与 EAR 维度混淆、牙区被遮，非许可问题）；Pexels `65665`（404）。

## 复核方式

三张均经 qa 目视核验（人数/正脸/清晰度）与 `file`+`sips` 格式校验；真实检出数以 dev 侧
mediapipe/insightface 实测为准（工单 04 验收）。

# SimpleSLAM 主用法笔记

这份笔记记录当前最实用的运行方式：跑 LO demo、跑 KITTI 00、导出轨迹、可视化并和真值比较。

## 1. 构建

```bash
cd /root/gpufree-data/SimpleSLAM

cmake -B build -DBUILD_TESTING=ON
cmake --build build -j
```

确认测试通过：

```bash
ctest --test-dir build --output-on-failure
```

## 2. 不用数据集：合成点云快速体验

```bash
./build/demo/run_lo_demo --synthetic --max-frames 20 \
    --output-prefix KITTI/results/lo_demo_synthetic
```

输出：

```text
KITTI/results/lo_demo_synthetic.kitti.txt
KITTI/results/lo_demo_synthetic.tum.txt
```

合成 demo 会模拟雷达沿 x 方向移动，输出里的 `tx` 应该逐帧增长。

## 3. KITTI 00 数据结构

当前 demo 使用 KITTI odometry sequence 目录：

```text
KITTI/dataset/sequences/00/
├── calib.txt
├── times.txt
└── velodyne/
    ├── 000000.bin
    ├── 000001.bin
    └── ...
```

真值轨迹在：

```text
KITTI/dataset/poses/00.txt
```

检查帧数：

```bash
find KITTI/dataset/sequences/00/velodyne -maxdepth 1 -type f -name '*.bin' | wc -l
wc -l KITTI/dataset/poses/00.txt
```

## 4. KITTI 00 前 100 帧：较准但慢

这个参数用原始点更多，前 100 帧能看较好的轨迹，但速度慢。

```bash
./build/demo/run_lo_demo --kitti ./KITTI/dataset/sequences/00 \
    --max-frames 100 \
    --output-prefix KITTI/results/kitti00_lo
```

已观测速度：约 100 帧 607 秒。

输出：

```text
KITTI/results/kitti00_lo.kitti.txt
KITTI/results/kitti00_lo.tum.txt
```

## 5. KITTI 00 快速版：适合交互学习

每帧抽点并减少 ICP 迭代：

```bash
./build/demo/run_lo_demo --kitti ./KITTI/dataset/sequences/00 \
    --max-frames 100 \
    --point-stride 5 \
    --max-points 8000 \
    --max-iterations 6 \
    --downsample-voxel 1.5 \
    --map-voxel 1.5 \
    --output-prefix KITTI/results/kitti00_lo_fast
```

参数含义：

- `--point-stride 5`：原始点每 5 个取 1 个。
- `--max-points 8000`：每帧最多使用 8000 个点。
- `--max-iterations 6`：每帧 ICP 最多迭代 6 次。
- `--downsample-voxel 1.5`：配准源点云体素降采样尺寸。
- `--map-voxel 1.5`：体素地图尺寸。

## 6. 跑完整 KITTI 00

`00` 序列约 4541 帧。不要用原始慢参数跑完整序列。

推荐完整快跑命令：

```bash
./build/demo/run_lo_demo --kitti ./KITTI/dataset/sequences/00 \
    --max-frames 0 \
    --point-stride 8 \
    --max-points 5000 \
    --max-iterations 5 \
    --downsample-voxel 2.0 \
    --map-voxel 2.0 \
    --print-every 100 \
    --output-prefix KITTI/results/kitti00_lo_full_fast
```

说明：

- `--max-frames 0`：跑完整序列。
- `--print-every 100`：每 100 帧打印一次，避免刷屏。
- 粗估耗时：几十分钟级别，取决于机器负载。

输出：

```text
KITTI/results/kitti00_lo_full_fast.kitti.txt
KITTI/results/kitti00_lo_full_fast.tum.txt
```

## 7. 可视化：SLAM 与真值轨迹

脚本：

```text
scripts/plot_trajectory.py
```

前 100 帧可视化：

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache python3 scripts/plot_trajectory.py \
    KITTI/dataset/poses/00.txt \
    KITTI/results/kitti00_lo_fast.kitti.txt \
    --out-prefix KITTI/results/kitti00_lo_fast_se3 \
    --align se3
```

完整序列可视化：

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache python3 scripts/plot_trajectory.py \
    KITTI/dataset/poses/00.txt \
    KITTI/results/kitti00_lo_full_fast.kitti.txt \
    --out-prefix KITTI/results/kitti00_lo_full_fast_se3 \
    --align se3
```

输出：

```text
*.trajectory.png
*.error.png
```

说明：

- `trajectory.png`：KITTI 真值轨迹和 LO 估计轨迹的俯视对比。
- `error.png`：逐帧平移误差。
- 图里画的是 KITTI 常用的 `x-z` 平面；在 KITTI 相机坐标系中，`z` 是前进方向，`y` 更接近竖直方向。

## 8. 精度验证

推荐先用可视化脚本输出的统计结果：

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache python3 scripts/plot_trajectory.py \
    KITTI/dataset/poses/00.txt \
    KITTI/results/kitti00_lo.kitti.txt \
    --out-prefix KITTI/results/kitti00_lo_se3 \
    --align se3
```

它会打印：

```text
frames
align
rmse
mean
median
max
std
```

已经观测到的前 100 帧较准版结果：

```text
SE3 align:
ATE RMSE ~= 0.305 m
mean     ~= 0.240 m
max      ~= 1.283 m

Sim3 align:
ATE RMSE ~= 0.179 m
mean     ~= 0.127 m
max      ~= 0.881 m
```

如果安装了 `evo`，也可以用项目自带脚本：

```bash
head -n 100 KITTI/dataset/poses/00.txt > KITTI/results/kitti00_gt_100.kitti.txt

python3 scripts/evaluate.py \
    KITTI/results/kitti00_gt_100.kitti.txt \
    KITTI/results/kitti00_lo.kitti.txt \
    --format kitti \
    --metric both
```

注意：

- KITTI 真值 `poses/00.txt` 是相机坐标系轨迹。
- 当前 LO demo 主要用于学习，输出轨迹通过整体对齐后比较更合理。
- 如果要严格评测，需要结合 `calib.txt` 里的外参做 LiDAR/camera 坐标转换。

## 9. 当前理解

这个项目当前已经能跑通：

```text
KITTI velodyne/*.bin
-> KittiSource
-> LoIcpOdometry
-> VoxelHashTarget + IcpSolver
-> trajectory
-> KITTI/TUM 轨迹文件
-> 可视化与误差统计
```

但是它仍然是学习版 LO：

- 原始点跑得慢。
- 快速版依赖抽点和较粗体素。
- 适合理解 SLAM 前端流程，不是高性能工程实现。

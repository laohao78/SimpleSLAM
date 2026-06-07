#include <SimpleSLAM/core/infra/topic_hub.hpp>
#include <SimpleSLAM/odometry/lo_icp_odometry.hpp>
#include <SimpleSLAM/resources/trajectory.hpp>
#include <SimpleSLAM/sensor_io/kitti_source.hpp>

#include <Eigen/Core>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace simpleslam;

struct Args {
    std::optional<std::string> kitti_path;
    size_t max_frames = 20;
    size_t point_stride = 1;
    size_t max_points = 0;
    int max_iterations = 20;
    float downsample_voxel_size = 1.0f;
    float map_voxel_size = 1.0f;
    size_t print_every = 1;
    std::string output_prefix = "KITTI/results/lo_demo_trajectory";
    bool help = false;
};

class SyntheticLidarSource final : public ISensorSource {
public:
    SyntheticLidarSource(size_t frames, double step_m)
        : total_frames_(frames), step_m_(step_m), world_points_(makeWorldPoints()) {}

    [[nodiscard]] bool hasNext() const override {
        return current_index_ < total_frames_;
    }

    std::optional<LidarScan> nextScan() override {
        if (!hasNext()) return std::nullopt;

        const double tx = static_cast<double>(current_index_) * step_m_;
        LidarScan scan;
        scan.timestamp = static_cast<Timestamp>(current_index_) * 0.1;
        scan.points.reserve(world_points_.size());

        const Eigen::Vector3f sensor_position(static_cast<float>(tx), 0.0f, 0.0f);
        for (const auto& p_world : world_points_) {
            const Eigen::Vector3f p_sensor = p_world - sensor_position;
            if (p_sensor.norm() < 80.0f) {
                scan.points.push_back(p_sensor);
            }
        }

        ++current_index_;
        return scan;
    }

    [[nodiscard]] Timestamp currentTimestamp() const override {
        return static_cast<Timestamp>(current_index_) * 0.1;
    }

private:
    static std::vector<Eigen::Vector3f> makeWorldPoints() {
        std::vector<Eigen::Vector3f> points;
        points.reserve(2000);

        for (int x = -20; x <= 50; x += 2) {
            for (int y = -10; y <= 10; y += 2) {
                for (int z = -2; z <= 3; ++z) {
                    const float wiggle = 0.15f * static_cast<float>((x + 2 * y + 3 * z) % 5);
                    points.emplace_back(static_cast<float>(x) + wiggle,
                                        static_cast<float>(y),
                                        static_cast<float>(z));
                }
            }
        }

        return points;
    }

    size_t total_frames_;
    double step_m_;
    std::vector<Eigen::Vector3f> world_points_;
    size_t current_index_{0};
};

void printUsage(std::string_view program) {
    std::cout
        << "Usage:\n"
        << "  " << program << " [--synthetic] [--max-frames N] [--output-prefix PATH]\n"
        << "  " << program << " --kitti /path/to/KITTI/sequences/00 [--max-frames N]\n"
        << "  " << program << " /path/to/KITTI/sequences/00\n\n"
        << "Options:\n"
        << "  --point-stride N          Keep every Nth raw point before odometry (default: 1)\n"
        << "  --max-points N            Keep at most N raw points per scan after stride (0 = no limit)\n"
        << "  --downsample-voxel SIZE   ICP source voxel size in meters (default: 1.0)\n"
        << "  --map-voxel SIZE          VoxelHashTarget map voxel size in meters (default: 1.0)\n"
        << "  --max-iterations N        ICP iterations per frame (default: 20)\n\n"
        << "  --print-every N           Print one row every N frames (default: 1)\n\n"
        << "Examples:\n"
        << "  " << program << " --synthetic --max-frames 20\n"
        << "  " << program << " --kitti /data/kitti/sequences/00 --max-frames 100 \\\n"
        << "      --point-stride 5 --max-iterations 8 --output-prefix KITTI/results/kitti00_fast\n";
}

Args parseArgs(int argc, char** argv) {
    Args args;

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);

        if (arg == "--help" || arg == "-h") {
            args.help = true;
        } else if (arg == "--synthetic") {
            args.kitti_path.reset();
        } else if (arg == "--kitti") {
            if (i + 1 >= argc) throw std::runtime_error("--kitti requires a path");
            args.kitti_path = argv[++i];
        } else if (arg == "--max-frames") {
            if (i + 1 >= argc) throw std::runtime_error("--max-frames requires a number");
            args.max_frames = static_cast<size_t>(std::stoull(argv[++i]));
        } else if (arg == "--point-stride") {
            if (i + 1 >= argc) throw std::runtime_error("--point-stride requires a number");
            args.point_stride = static_cast<size_t>(std::stoull(argv[++i]));
            if (args.point_stride == 0) throw std::runtime_error("--point-stride must be > 0");
        } else if (arg == "--max-points") {
            if (i + 1 >= argc) throw std::runtime_error("--max-points requires a number");
            args.max_points = static_cast<size_t>(std::stoull(argv[++i]));
        } else if (arg == "--downsample-voxel") {
            if (i + 1 >= argc) throw std::runtime_error("--downsample-voxel requires a number");
            args.downsample_voxel_size = std::stof(argv[++i]);
            if (args.downsample_voxel_size <= 0.0f) {
                throw std::runtime_error("--downsample-voxel must be > 0");
            }
        } else if (arg == "--map-voxel") {
            if (i + 1 >= argc) throw std::runtime_error("--map-voxel requires a number");
            args.map_voxel_size = std::stof(argv[++i]);
            if (args.map_voxel_size <= 0.0f) throw std::runtime_error("--map-voxel must be > 0");
        } else if (arg == "--max-iterations") {
            if (i + 1 >= argc) throw std::runtime_error("--max-iterations requires a number");
            args.max_iterations = std::stoi(argv[++i]);
            if (args.max_iterations <= 0) throw std::runtime_error("--max-iterations must be > 0");
        } else if (arg == "--print-every") {
            if (i + 1 >= argc) throw std::runtime_error("--print-every requires a number");
            args.print_every = static_cast<size_t>(std::stoull(argv[++i]));
            if (args.print_every == 0) throw std::runtime_error("--print-every must be > 0");
        } else if (arg == "--output-prefix") {
            if (i + 1 >= argc) throw std::runtime_error("--output-prefix requires a path");
            args.output_prefix = argv[++i];
        } else if (!arg.empty() && arg.front() != '-') {
            args.kitti_path = std::string(arg);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }

    return args;
}

LoIcpConfig makeDemoConfig(bool kitti_mode, const Args& args) {
    LoIcpConfig cfg;
    cfg.target.voxel_size = kitti_mode ? args.map_voxel_size : 0.8f;
    cfg.target.max_points_per_voxel = 30;
    cfg.target.max_range = kitti_mode ? 100.0 : 80.0;
    cfg.target.max_correspondence_dist = kitti_mode ? 3.0 : 2.0;
    cfg.solver.max_iterations = args.max_iterations;
    cfg.solver.convergence_threshold = 1e-5;
    cfg.downsample_voxel_size = kitti_mode ? args.downsample_voxel_size : 0.5f;
    return cfg;
}

LidarScan decimateScan(const LidarScan& input, size_t stride, size_t max_points) {
    if (stride <= 1 && (max_points == 0 || input.points.size() <= max_points)) {
        return input;
    }

    LidarScan output;
    output.timestamp = input.timestamp;
    output.layout.type = ScanLayout::Type::Unorganized;

    const size_t reserve_count =
        max_points > 0 ? std::min(max_points, (input.points.size() + stride - 1) / stride)
                       : (input.points.size() + stride - 1) / stride;
    output.points.reserve(reserve_count);

    const bool copy_intensity = input.intensities.has_value();
    if (copy_intensity) output.intensities.emplace().reserve(reserve_count);

    for (size_t i = 0; i < input.points.size(); i += stride) {
        if (max_points > 0 && output.points.size() >= max_points) break;
        output.points.push_back(input.points[i]);
        if (copy_intensity) output.intensities->push_back((*input.intensities)[i]);
    }

    return output;
}

std::string statusName(TrackingStatus status) {
    switch (status) {
    case TrackingStatus::Initializing:
        return "initializing";
    case TrackingStatus::Tracking:
        return "tracking";
    case TrackingStatus::Degraded:
        return "degraded";
    case TrackingStatus::Lost:
        return "lost";
    }
    return "unknown";
}

void createParentDirectory(const std::string& output_prefix) {
    const std::filesystem::path prefix(output_prefix);
    const auto parent = prefix.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parseArgs(argc, argv);
        if (args.help) {
            printUsage(argv[0]);
            return 0;
        }

        const bool kitti_mode = args.kitti_path.has_value();
        const size_t max_frames = args.max_frames == 0 ? std::numeric_limits<size_t>::max()
                                                       : args.max_frames;

        std::unique_ptr<ISensorSource> source;
        if (kitti_mode) {
            source = std::make_unique<KittiSource>(*args.kitti_path);
        } else {
            source = std::make_unique<SyntheticLidarSource>(max_frames, 0.2);
        }

        LoIcpOdometry odometry(makeDemoConfig(kitti_mode, args));
        Trajectory trajectory;

        TopicHub::init(true);
        odometry.initialize(TopicHub::instance());

        std::cout << "SimpleSLAM LO demo\n";
        std::cout << "source: " << (kitti_mode ? *args.kitti_path : "synthetic") << "\n";
        std::cout << "max_frames: " << args.max_frames << "\n\n";
        std::cout << "point_stride: " << args.point_stride << "\n";
        std::cout << "max_points: " << args.max_points << "\n";
        std::cout << "downsample_voxel: " << args.downsample_voxel_size << "\n";
        std::cout << "map_voxel: " << args.map_voxel_size << "\n";
        std::cout << "max_iterations: " << args.max_iterations << "\n\n";
        std::cout << "print_every: " << args.print_every << "\n\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "frame,timestamp,status,raw_points,used_points,tx,ty,tz\n";

        size_t frames_processed = 0;
        const auto start = std::chrono::steady_clock::now();

        while (source->hasNext() && frames_processed < max_frames) {
            auto scan = source->nextScan();
            if (!scan) continue;

            auto used_scan = decimateScan(*scan, args.point_stride, args.max_points);
            const auto result = odometry.processLidar(used_scan);
            TopicHub::instance().drainAll();

            if (result.status != TrackingStatus::Lost) {
                trajectory.append(result.timestamp, result.pose);
            }

            const auto& t = result.pose.translation();
            if (frames_processed % args.print_every == 0) {
                std::cout << frames_processed << ","
                          << result.timestamp << ","
                          << statusName(result.status) << ","
                          << scan->size() << ","
                          << used_scan.size() << ","
                          << t.x() << ","
                          << t.y() << ","
                          << t.z() << "\n";
            }

            ++frames_processed;
        }

        odometry.shutdown();
        TopicHub::shutdown();

        const auto elapsed = std::chrono::steady_clock::now() - start;
        const double elapsed_seconds = std::chrono::duration<double>(elapsed).count();

        createParentDirectory(args.output_prefix);
        const std::string kitti_output = args.output_prefix + ".kitti.txt";
        const std::string tum_output = args.output_prefix + ".tum.txt";
        trajectory.exportKITTI(kitti_output);
        trajectory.exportTUM(tum_output);

        std::cout << "\nframes_processed: " << frames_processed << "\n";
        std::cout << "trajectory_poses: " << trajectory.size() << "\n";
        std::cout << "elapsed_seconds: " << elapsed_seconds << "\n";
        std::cout << "wrote: " << kitti_output << "\n";
        std::cout << "wrote: " << tum_output << "\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "run_lo_demo error: " << e.what() << "\n\n";
        printUsage(argv[0]);
        return 1;
    }
}

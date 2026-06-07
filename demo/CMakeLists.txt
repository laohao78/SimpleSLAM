# Small runnable demos for learning and manual smoke tests.

add_executable(run_lo_demo
    run_lo_demo.cpp
)

target_compile_features(run_lo_demo PRIVATE cxx_std_20)

target_link_libraries(run_lo_demo
    PRIVATE
        simpleslam_sensor_io
        simpleslam_odometry
        simpleslam_resources
        simpleslam_compiler_options
)

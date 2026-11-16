#!/usr/bin/env bash
set -e
PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_ROOT="${1:-$PACKAGE_ROOT/../.castrack_interfaces}"
source /opt/ros/lyrical/setup.bash
colcon --log-base "$BUILD_ROOT/log" build \
  --base-paths "$PACKAGE_ROOT/ros2_interfaces/castrack_interfaces" \
  --build-base "$BUILD_ROOT/build" --install-base "$BUILD_ROOT/install" \
  --cmake-args -DBUILD_TESTING=OFF
printf '\nSource the generated ROS interface setup:\n%s\n' "$BUILD_ROOT/install/setup.bash"

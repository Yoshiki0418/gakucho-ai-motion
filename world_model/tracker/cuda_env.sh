# Sourced by the training scripts from the repository root.
# ホストの libcuda がカーネルのドライバと同じ版ならそのまま使う(2026-11-06 以降の 595.99.02)。
# 版がずれているとき(2026-10 のホスト)だけ、同梱の .runtime/nvidia-<版> と libcuda.so の shim を使う。
driver=$(sed -n 's/.*Module for x86_64 *\([0-9.]*\).*/\1/p' /proc/driver/nvidia/version)
if [ ! -e "/usr/lib/x86_64-linux-gnu/libcuda.so.$driver" ]; then
  export LD_LIBRARY_PATH="$PWD/.runtime/warp-cuda-shim:$PWD/.runtime/nvidia-$driver"
fi

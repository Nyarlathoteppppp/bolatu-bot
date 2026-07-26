#!/usr/bin/env bash
set -euo pipefail

install_path="${INSTALL_PATH:-/usr/local/bin/coscli}"
arch="$(uname -m)"
case "$arch" in
  x86_64|amd64)
    url="https://cosbrowser.cloud.tencent.com/software/coscli/coscli-linux-amd64"
    ;;
  aarch64|arm64)
    url="https://cosbrowser.cloud.tencent.com/software/coscli/coscli-linux-arm64"
    ;;
  *)
    echo "Unsupported architecture: $arch" >&2
    exit 2
    ;;
esac

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
wget -q -O "$tmp" "$url"
chmod 755 "$tmp"
if [[ -w "$(dirname "$install_path")" ]]; then
  mv "$tmp" "$install_path"
else
  sudo install -m 0755 "$tmp" "$install_path"
fi
"$install_path" --version

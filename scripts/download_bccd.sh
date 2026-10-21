#!/usr/bin/env bash
# Download the BCCD dataset (Shenggan/BCCD_Dataset) into data/.
set -euo pipefail

DEST="${1:-data}"
mkdir -p "$DEST"
URL="https://github.com/Shenggan/BCCD_Dataset/archive/refs/heads/master.tar.gz"
echo "Downloading BCCD dataset..."
curl -sL "$URL" -o "$DEST/bccd.tgz"
tar -xzf "$DEST/bccd.tgz" -C "$DEST"
rm -f "$DEST/bccd.tgz"
echo "Done -> $DEST/BCCD_Dataset-master"
echo "Images : $DEST/BCCD_Dataset-master/BCCD/JPEGImages"
echo "Labels : $DEST/BCCD_Dataset-master/BCCD/Annotations"

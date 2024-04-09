#!/usr/bin/env bash

TARGET_DIR="$1"
mkdir -p "$TARGET_DIR"

shift
for name in "$@" ; do
  URL="http://gem.cs.ru.nl/${name}.tar.gz"
  echo Fetching $URL
  curl "$URL" | tar xz --directory "$TARGET_DIR"
done



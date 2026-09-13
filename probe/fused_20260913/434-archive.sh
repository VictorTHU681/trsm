#!/usr/bin/env bash
set -eo pipefail
sha256sum generated/*.c generated/*.h results/*.done results/*.running worker.sh > source-sha256.txt
sha256sum generated/front-control.text generated/front64.text > binary-text-sha256.txt
cmp generated/front-control.text generated/front64.text
hostname
date -Iseconds
echo ARCHIVE_SOURCE_MANIFEST_READY
touch STOP

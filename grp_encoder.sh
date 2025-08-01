#!/bin/sh

if [ "$RUN_ID" = "24" ] || [ "$RUN_ID" = "25" ]; then
  export GRP_ENCODER=true
else
  export GRP_ENCODER=false
fi

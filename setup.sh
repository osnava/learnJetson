#!/bin/bash
# Install nano if not present
if ! command -v nano &> /dev/null; then
    apt-get update && apt-get install -y nano
fi

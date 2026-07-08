#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec streamlit run main.py --server.port 8080 --server.address 0.0.0.0

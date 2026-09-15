#!/usr/bin/env bash
set -e

echo "Configuring macOS XGBoost OpenMP runtime..."
VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Error: $VENV_DIR directory not found. Please create it first."
    exit 1
fi

SKLEARN_LIBOMP=$(find "$VENV_DIR" -name "libomp.dylib" | grep "sklearn" | head -n 1)
XGBOOST_LIB_DIR=$(find "$VENV_DIR" -type d -path "*/xgboost/lib" | head -n 1)

if [ -n "$SKLEARN_LIBOMP" ] && [ -n "$XGBOOST_LIB_DIR" ]; then
    echo "Copying bundled OpenMP: $SKLEARN_LIBOMP -> $XGBOOST_LIB_DIR/"
    cp "$SKLEARN_LIBOMP" "$XGBOOST_LIB_DIR/"
    echo "Updating LC_RPATH on libxgboost.dylib..."
    install_name_tool -add_rpath "@loader_path" "$XGBOOST_LIB_DIR/libxgboost.dylib" 2>/dev/null || true
    echo "macOS XGBoost configured successfully!"
else
    echo "Warning: could not locate bundled libomp.dylib or xgboost/lib."
fi


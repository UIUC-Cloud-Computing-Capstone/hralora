#!/bin/bash

# Simple Flower Runner Script
# Quick start for testing with minimal clients

set -e

# Default values
NUM_CLIENTS=${1:-3}
NUM_ROUNDS=${2:-5}

echo "Starting Flower with $NUM_CLIENTS clients for $NUM_ROUNDS rounds..."

# Cleanup existing processes
pkill -f "python.*flower" 2>/dev/null || true
sleep 2

# Start server
echo "Starting server..."
python flower_server.py --num_rounds $NUM_ROUNDS --log_level INFO &
SERVER_PID=$!

# Wait for server to start
sleep 3

# Start clients
echo "Starting $NUM_CLIENTS clients..."
for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    echo "Starting client $i..."
    python flower_client.py --client_id $i --log_level INFO &
    sleep 1
done

echo "All processes started. Press Ctrl+C to stop."
echo "Server PID: $SERVER_PID"

# Wait for user interrupt
trap 'echo "Stopping..."; pkill -f "python.*flower"; exit 0' INT

# Wait for server to finish
wait $SERVER_PID

echo "Training completed!"

#!/bin/bash
# Resource Monitoring Script for City OS
# Logs CPU, RAM, and GPU metrics during active streaming

LOG_FILE="surveillance_hardware_metrics.log"
INTERVAL=2

echo "========================================================================="
echo "   SURVEILLANCE WORKLOAD HEALTH MONITORING SYSTEM"
echo "   Logging data to: $LOG_FILE every $INTERVAL seconds."
echo "========================================================================="
echo "Timestamp, CPU_Util%, System_RAM_Used/Total, GPU_VRAM_Used/Total, GPU_Util%" > "$LOG_FILE"

while true; do
    TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
    
    # 1. CPU utilization
    CPU_UTIL=$(top -bn1 | grep "Cpu(s)" | awk '{print 100 - $8}' 2>/dev/null || echo "0")
    
    # 2. System RAM
    SYS_RAM=$(free -m 2>/dev/null | awk '/Mem:/ {print $3 "/" $2 " MB"}' || echo "N/A")
    
    # 3. GPU metrics (if nvidia-smi available)
    if command -v nvidia-smi &> /dev/null; then
        GPU_METRICS=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo "0,0,0")
        GPU_VRAM=$(echo "$GPU_METRICS" | awk -F',' '{print $1 "/" $2 " MB"}')
        GPU_UTIL=$(echo "$GPU_METRICS" | awk -F',' '{print $3 "%"}')
    else
        GPU_VRAM="N/A (No CUDA)"
        GPU_UTIL="N/A"
    fi

    # Output to stdout and log
    OUTPUT_STRING="[$TIMESTAMP] CPU: $CPU_UTIL% | Sys RAM: $SYS_RAM | GPU VRAM: $GPU_VRAM | GPU Util: $GPU_UTIL"
    echo "$OUTPUT_STRING"
    echo "$TIMESTAMP, $CPU_UTIL, $SYS_RAM, $GPU_VRAM, $GPU_UTIL" >> "$LOG_FILE"
    
    # Health warnings
    CPU_NUM=$(echo "$CPU_UTIL" | awk '{print int($1+0.5)}')
    if [ "$CPU_NUM" -gt 90 ]; then
        echo "  ⚠ WARNING: CPU > 90% - Risk of processing bottleneck!"
    fi
    
    RAM_USED_MB=$(echo "$SYS_RAM" | awk -F'/' '{print $1}' | awk '{print int($1+0.5)}')
    RAM_TOTAL_MB=$(echo "$SYS_RAM" | awk -F'/' '{print $2}' | awk '{print int($1+0.5)}')
    if [ "$RAM_TOTAL_MB" -gt 0 ]; then
        RAM_PCT=$((RAM_USED_MB * 100 / RAM_TOTAL_MB))
        if [ "$RAM_PCT" -gt 95 ]; then
            echo "  ⚠ WARNING: RAM > 95% - Risk of OOM!"
        fi
    fi
    
    if [ "$GPU_UTIL" != "N/A" ]; then
        GPU_NUM=$(echo "$GPU_UTIL" | awk -F'%' '{print int($1+0.5)}')
        if [ "$GPU_NUM" -eq 100 ]; then
            echo "  ⚠ WARNING: GPU at 100% - Consider frame skipping!"
        fi
    fi
    
    sleep "$INTERVAL"
done
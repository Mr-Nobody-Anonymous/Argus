"""
Headless WebSocket Stress Tester for City OS
Floods backend with simulated asset tracking updates
"""
import asyncio
import random
import time
import json

# Try to import websockets, fallback to simulation
try:
    import websockets
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False

WS_URL = "ws://localhost:8000/ws/telemetry"
NUM_ASSETS = 100
UPDATES_PER_SECOND = 100

# Base anchor coordinates
LAT_BASE, LON_BASE = 8.54, 39.27

async def simulate_asset_updates():
    """Flood backend with 100 concurrent asset updates per second."""
    if not WS_AVAILABLE:
        print("WARNING: websockets library not installed. Running simulation mode.")
        return await run_simulation()
    
    try:
        async with websockets.connect(WS_URL) as websocket:
            print(f"Connected to {WS_URL}")
            print(f"Flooding {UPDATES_PER_SECOND} tracking updates/sec across {NUM_ASSETS} assets...")
            
            interval = 1.0 / UPDATES_PER_SECOND
            updates_sent = 0
            
            try:
                while True:
                    t_start = time.time()
                    
                    for _ in range(10):  # Send 10 updates per iteration
                        payload = {
                            "asset_id": f"ASSET_{random.randint(1, NUM_ASSETS):03d}",
                            "latitude": round(LAT_BASE + random.uniform(-0.01, 0.01), 6),
                            "longitude": round(LON_BASE + random.uniform(-0.01, 0.01), 6),
                            "velocity": round(random.uniform(10, 80), 2),
                            "timestamp": time.time()
                        }
                        
                        await websocket.send(json.dumps(payload))
                        updates_sent += 1
                    
                    # Dynamic pacing
                    elapsed = time.time() - t_start
                    await asyncio.sleep(max(0, interval - elapsed))
                    
                    if updates_sent % 1000 == 0:
                        print(f"Sent {updates_sent} updates...")
                        
            except KeyboardInterrupt:
                print(f"\nStress tester halted. Total updates sent: {updates_sent}")
                
    except Exception as e:
        print(f"Connection error: {e}")

async def run_simulation():
    """Simulation mode for testing without backend connection."""
    print(f"Running simulation mode...")
    
    updates_sent = 0
    interval = 1.0 / UPDATES_PER_SECOND
    
    try:
        while True:
            t_start = time.time()
            
            for _ in range(10):
                payload = {
                    "asset_id": f"ASSET_{random.randint(1, NUM_ASSETS):03d}",
                    "latitude": round(LAT_BASE + random.uniform(-0.01, 0.01), 6),
                    "longitude": round(LON_BASE + random.uniform(-0.01, 0.01), 6),
                    "velocity": round(random.uniform(10, 80), 2),
                    "timestamp": time.time()
                }
                # Simulate send
                updates_sent += 1
            
            elapsed = time.time() - t_start
            time.sleep(max(0, interval - elapsed))
            
            if updates_sent % 500 == 0:
                print(f"[SIM] Would have sent {updates_sent} updates to backend")
                
    except KeyboardInterrupt:
        print(f"\nSimulation halted. Total simulated updates: {updates_sent}")

if __name__ == "__main__":
    try:
        asyncio.run(simulate_asset_updates())
    except KeyboardInterrupt:
        print("\nStopped.")
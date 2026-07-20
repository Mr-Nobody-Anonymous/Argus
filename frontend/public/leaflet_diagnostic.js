/**
 * Frontend Canvas Render Diagnostic Tool
 * Inject this script into your browser's Developer Tools Console (F12)
 * on your Leaflet map page to monitor performance
 */
(function runFrontendStressDiagnostic() {
    console.log("Initializing Canvas Render Diagnostic monitor...");

    let frameCount = 0;
    let lastTime = performance.now();
    let fpsLog = [];
    let droppedFrames = 0;
    
    // Track browser heap sizes
    function getMemoryUsage() {
        return window.performance && window.performance.memory ? 
            (window.performance.memory.usedJSHeapSize / (1024 * 1024)).toFixed(2) + " MB" : "N/A";
    }

    // Monitor render loop
    function monitorRenderLoop() {
        frameCount++;
        let now = performance.now();
        let delta = now - lastTime;

        if (delta >= 1000) {
            let currentFPS = Math.round((frameCount * 1000) / delta);
            fpsLog.push(currentFPS);
            console.log(`[UI DIAGNOSTIC] Render Performance: ${currentFPS} FPS | Heap: ${getMemoryUsage()}`);
            
            // SUCCESS CRITERIA: FPS > 30
            if (currentFPS < 30) {
                droppedFrames++;
                console.warn(`[PERFORMANCE CRITICAL] Frame Drop! FPS: ${currentFPS}`);
            }
            
            // Memory leak detection
            const memoryMB = parseFloat(getMemoryUsage());
            if (memoryMB > 500 && window.performance.memory) {
                console.warn(`[MEMORY WARNING] Heap size: ${memoryMB}MB - Potential leak!`);
            }
            
            frameCount = 0;
            lastTime = now;
        }
        
        requestAnimationFrame(monitorRenderLoop);
    }
    
    // Start monitoring
    const monitorId = requestAnimationFrame(monitorRenderLoop);
    
    // Expose for stopping
    window.stopDiagnostics = function() {
        cancelAnimationFrame(monitorId);
        console.log("[DIAGNOSTIC] Stopped. Summary:", {
            avgFPS: fpsLog.length ? fpsLog.reduce((a,b) => a+b, 0) / fpsLog.length : 0,
            totalDropped: droppedFrames
        });
    };
    
    console.log("Diagnostic running. Call stopDiagnostics() to stop.");
})();

/**
 * Leaflet Layer Performance Checker
 * Tests map rendering under load
 */
function checkLeafletLayers() {
    if (!window.L) {
        console.error("Leaflet not loaded!");
        return;
    }
    
    const map = Object.values(window).find(v => v && v._container && v._container.classList);
    
    if (map) {
        const canvasLayers = map.getCanvas && typeof map.getCanvas === 'function' ? 
            map.getCanvas().getElementsByTagName('canvas').length : 'unknown';
        console.log(`[LEAFLET CHECK] Active canvas layers: ${canvasLayers}`);
        
        // Check for memory leaks in marker clusters
        const markers = map._layers ? Object.keys(map._layers).length : 0;
        console.log(`[LEAFLET CHECK] Total map objects: ${markers}`);
    }
}
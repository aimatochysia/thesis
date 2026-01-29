const SAMPLING_RATE = 360;
const DISPLAY_SECONDS = 5;
const DISPLAY_SAMPLES = SAMPLING_RATE * DISPLAY_SECONDS;
const TARGET_FPS = 60;

let ecgData = [];
let annotations = [];
let currentIndex = 0;
let isRunning = false;
let animationId = null;
let classifications = [];
let falseDetections = [];
let speedMultiplier = 1;
let currentBeatWaveform = null;
let currentRPeakPos = 70;
let viewOffset = 0;
let isLive = true;
let lastFrameTime = 0;

let ecgRenderer = null;
let beatRenderer = null;

document.addEventListener('DOMContentLoaded', async () => {
    ecgRenderer = new ECGRenderer('ecgCanvas');
    beatRenderer = new BeatRenderer('beatCanvas');
    
    ecgRenderer.setDragCallback((deltaSeconds) => {
        if (currentIndex < DISPLAY_SAMPLES) return;
        
        viewOffset += deltaSeconds;
        const maxHistory = -currentIndex / SAMPLING_RATE;
        viewOffset = Math.max(maxHistory, Math.min(0, viewOffset));
        isLive = viewOffset >= -0.1;
        
        updateHistoryUI();
        drawECG();
        updateTime();
    });
    
    await loadModelInfo();
    await loadData();
    drawECG();
    
    console.log('ECG Real-Time Classification App initialized');
});

async function loadData() {
    try {
        console.log('[ECG] Loading ECG data from backend...');
        const data = await api.loadData();
        ecgData = data.signal;
        annotations = data.annotations;
        console.log(`[ECG] ✓ Loaded ${ecgData.length} samples and ${annotations.length} annotations`);
        console.log(`[ECG] First annotation:`, annotations[0]);
        console.log(`[ECG] Signal min/max:`, Math.min(...ecgData.slice(0, 1000)), Math.max(...ecgData.slice(0, 1000)));
    } catch (error) {
        console.error('[ECG] ✗ Failed to load ECG data:', error);
        showError('Failed to load ECG data. Is the backend running?');
    }
}

async function loadModelInfo() {
    try {
        const status = await api.getStatus();
        document.getElementById('modelName').textContent = status.model.name;
        
        const beatLength = status.model.beat_length || 188;
        document.getElementById('beatSamplesInfo').textContent = 
            `${beatLength} samples extracted around R-peak`;
    } catch (error) {
        console.error('Failed to load model info:', error);
        document.getElementById('modelName').textContent = 'Error';
    }
}

function showError(message) {
    console.error(message);
    alert(message);
}

function setSpeed(speed) {
    speedMultiplier = speed;
    document.getElementById('speedValue').textContent = speed + 'x';
    
    document.querySelectorAll('.speed-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.textContent === speed + 'x') {
            btn.classList.add('active');
        }
    });
    
    api.control('set_speed', { speed }).catch(console.error);
}

function scrollHistory(seconds) {
    if (currentIndex < DISPLAY_SAMPLES) return;
    
    viewOffset += seconds;
    const maxHistory = -currentIndex / SAMPLING_RATE;
    viewOffset = Math.max(maxHistory, Math.min(0, viewOffset));
    isLive = viewOffset >= -0.1;
    
    updateHistoryUI();
    drawECG();
    updateTime();
}

function goToLive() {
    viewOffset = 0;
    isLive = true;
    updateHistoryUI();
    drawECG();
    updateTime();
}

function navigateToTime(sampleIndex) {
    const targetOffset = (sampleIndex - currentIndex + DISPLAY_SAMPLES / 2) / SAMPLING_RATE;
    if (targetOffset >= 0) {
        goToLive();
        return;
    }
    viewOffset = targetOffset;
    isLive = false;
    updateHistoryUI();
    drawECG();
    updateTime();
}

function updateHistoryUI() {
    const indicator = document.getElementById('historyIndicator');
    const fwdBtn = document.getElementById('fwdBtn');
    const fwd5Btn = document.getElementById('fwd5Btn');
    
    indicator.style.display = isLive ? 'none' : 'inline';
    fwdBtn.disabled = isLive;
    fwd5Btn.disabled = isLive;
}

function drawECG() {
    let endSample = isLive ? currentIndex : Math.max(0, currentIndex + Math.round(viewOffset * SAMPLING_RATE));
    let startSample = Math.max(0, endSample - DISPLAY_SAMPLES);
    
    const samples = ecgData.slice(startSample, endSample);
    
    ecgRenderer.render({
        samples,
        annotations,
        startSample,
        classifications,
        isLive
    });
}

function drawBeatWaveform(waveform, isAbnormal = false) {
    beatRenderer.render({
        waveform,
        rPeakPos: currentRPeakPos,
        isAbnormal
    });
}

function updateTime() {
    let displayIndex = isLive ? currentIndex : Math.max(0, currentIndex + Math.round(viewOffset * SAMPLING_RATE));
    const seconds = displayIndex / SAMPLING_RATE;
    const minutes = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(3);
    document.getElementById('currentTime').textContent = `${minutes}:${secs.padStart(6, '0')}`;
}

async function checkForBeats() {
    const samplesToCheck = Math.max(1, Math.round(speedMultiplier * (SAMPLING_RATE / TARGET_FPS)));
    const prevSample = currentIndex - samplesToCheck;
    
    for (const ann of annotations) {
        if (ann.sample_index > prevSample && ann.sample_index <= currentIndex && ann.beat_type !== '+') {
            try {
                console.log(`[ECG] Calling model for beat at sample ${ann.sample_index}, type: ${ann.beat_type}`);
                const result = await api.classify(ann.sample_index, ann.beat_type);
                console.log(`[ECG] Classification result:`, result);
                if (result.predicted !== 'WAITING') {
                    addClassification(result);
                }
            } catch (error) {
                console.error('[ECG] Classification error:', error);
            }
        }
    }
}

function addClassification(result) {
    classifications.unshift(result);
    
    if (result.correct === false) {
        falseDetections.unshift(result);
        updateFalseDetectionList();
    }
    
    updateStats();
    updateCurrentStatus(result);
    updateBeatSnapshot(result);
    updateClassificationList(result);
}

function updateStats() {
    const total = classifications.length;
    const normal = classifications.filter(c => c.predicted === 'NORMAL').length;
    const abnormal = classifications.filter(c => c.predicted === 'ABNORMAL').length;
    const correct = classifications.filter(c => c.correct === true).length;
    
    document.getElementById('totalBeats').textContent = total;
    document.getElementById('normalBeats').textContent = normal;
    document.getElementById('abnormalBeats').textContent = abnormal;
    document.getElementById('falseCount').textContent = falseDetections.length;
    
    if (total > 0) {
        document.getElementById('accuracy').textContent = Math.round((correct / total) * 100) + '%';
    }
}

function updateCurrentStatus(result) {
    const statusEl = document.getElementById('currentStatus');
    statusEl.textContent = result.predicted;
    statusEl.className = 'value ' + result.predicted.toLowerCase();
}

function updateBeatSnapshot(result) {
    if (result.beat_waveform) {
        currentBeatWaveform = result.beat_waveform;
        currentRPeakPos = result.r_peak_pos_in_beat || 70;
        drawBeatWaveform(result.beat_waveform, result.predicted === 'ABNORMAL');
        
        document.getElementById('beatTypeDisplay').textContent = result.beat_type || '--';
        document.getElementById('beatTypeDisplay').style.color = (result.beat_type === 'N') ? '#00ff88' : '#ff4757';
        
        document.getElementById('groundTruthDisplay').textContent = result.ground_truth || '--';
        document.getElementById('groundTruthDisplay').style.color = (result.ground_truth === 'NORMAL') ? '#00ff88' : '#ff4757';
        
        document.getElementById('predictionDisplay').textContent = result.predicted;
        document.getElementById('predictionDisplay').style.color = (result.predicted === 'NORMAL') ? '#00ff88' : '#ff4757';
    }
}

function updateClassificationList(result) {
    const listEl = document.getElementById('classificationList');
    
    if (classifications.length === 1) {
        listEl.innerHTML = '';
    }
    
    const time = (result.r_peak / SAMPLING_RATE).toFixed(2);
    const item = document.createElement('div');
    item.className = 'classification-item ' + result.predicted.toLowerCase();
    if (result.correct === false) {
        item.classList.add('false');
    }
    item.onclick = () => navigateToTime(result.r_peak);
    item.innerHTML = `
        <div class="beat-info">
            <div>Beat Type: ${result.beat_type || '?'} → ${result.predicted}</div>
            <div class="beat-time">Time: ${time}s | Prob: ${(result.probability * 100).toFixed(1)}%</div>
        </div>
        <span class="prediction-badge ${result.predicted.toLowerCase()}">${result.predicted}</span>
    `;
    listEl.insertBefore(item, listEl.firstChild);
    
    while (listEl.children.length > 100) {
        listEl.removeChild(listEl.lastChild);
    }
}

function updateFalseDetectionList() {
    const listEl = document.getElementById('falseDetectionList');
    
    if (falseDetections.length === 0) {
        listEl.innerHTML = '<p class="placeholder-text">No false detections yet.</p>';
        return;
    }
    
    listEl.innerHTML = '';
    falseDetections.slice(0, 50).forEach(result => {
        const time = (result.r_peak / SAMPLING_RATE).toFixed(2);
        const item = document.createElement('div');
        item.className = 'false-detection-item';
        item.onclick = () => navigateToTime(result.r_peak);
        item.innerHTML = `
            <div>
                <span class="false-time">${time}s</span>
                <span class="false-details">Expected: ${result.ground_truth} | Got: ${result.predicted}</span>
            </div>
        `;
        listEl.appendChild(item);
    });
}

function animate(timestamp) {
    if (!isRunning) return;
    
    const frameInterval = 1000 / TARGET_FPS;
    const deltaTime = timestamp - lastFrameTime;
    
    if (deltaTime >= frameInterval) {
        lastFrameTime = timestamp - (deltaTime % frameInterval);
        
        const samplesPerSecond = SAMPLING_RATE * speedMultiplier;
        const samplesToAdvance = Math.max(1, Math.round(samplesPerSecond / TARGET_FPS));
        
        for (let i = 0; i < samplesToAdvance; i++) {
            if (currentIndex < ecgData.length) {
                currentIndex++;
            }
        }
        
        if (isLive) {
            drawECG();
            updateTime();
        }
        
        checkForBeats();
    }
    
    if (currentIndex < ecgData.length) {
        animationId = requestAnimationFrame(animate);
    } else {
        isRunning = false;
        document.getElementById('currentStatus').textContent = 'Complete!';
        document.getElementById('currentStatus').className = 'value';
    }
}

async function startSimulation() {
    if (ecgData.length === 0) {
        await loadData();
    }
    
    isRunning = true;
    lastFrameTime = performance.now();
    animationId = requestAnimationFrame(animate);
    
    api.control('start').catch(console.error);
}

function stopSimulation() {
    isRunning = false;
    if (animationId) {
        cancelAnimationFrame(animationId);
    }
    
    api.control('stop').catch(console.error);
}

function resetSimulation() {
    stopSimulation();
    
    currentIndex = 0;
    classifications = [];
    falseDetections = [];
    currentBeatWaveform = null;
    viewOffset = 0;
    isLive = true;
    
    document.getElementById('totalBeats').textContent = '0';
    document.getElementById('normalBeats').textContent = '0';
    document.getElementById('abnormalBeats').textContent = '0';
    document.getElementById('accuracy').textContent = '--';
    document.getElementById('heartRate').textContent = '--';
    document.getElementById('falseCount').textContent = '0';
    document.getElementById('currentStatus').textContent = 'Waiting...';
    document.getElementById('currentStatus').className = 'value waiting';
    document.getElementById('classificationList').innerHTML = '<p class="placeholder-text">No classifications yet. Start the simulation!</p>';
    document.getElementById('falseDetectionList').innerHTML = '<p class="placeholder-text">No false detections yet.</p>';
    document.getElementById('currentTime').textContent = '0:00.000';
    document.getElementById('beatTypeDisplay').textContent = '--';
    document.getElementById('groundTruthDisplay').textContent = '--';
    document.getElementById('predictionDisplay').textContent = '--';
    
    updateHistoryUI();
    
    beatRenderer.clear();
    beatRenderer.drawGrid();
    
    drawECG();
    
    api.control('reset').catch(console.error);
}

function exportECG(format = 'png') {
    let endSample = isLive ? currentIndex : Math.max(0, currentIndex + Math.round(viewOffset * SAMPLING_RATE));
    let startSample = Math.max(0, endSample - DISPLAY_SAMPLES);
    
    const samples = ecgData.slice(startSample, endSample);
    const modelName = document.getElementById('modelName').textContent || 'ECG Model';
    
    const exportOptions = {
        samples,
        annotations,
        startSample,
        classifications,
        modelName,
        timestamp: new Date().toISOString(),
        format,
        showGrid: true
    };
    
    const filename = `ecg_report_${new Date().toISOString().replace(/[:.]/g, '-')}`;
    ecgRenderer.downloadAsImage(exportOptions, filename);
}

function exportBeatSnapshot(format = 'png') {
    if (!currentBeatWaveform) {
        alert('No beat waveform available. Run the simulation first.');
        return;
    }
    
    const canvas = document.getElementById('beatCanvas');
    const dataUrl = canvas.toDataURL(`image/${format}`, 0.95);
    
    const link = document.createElement('a');
    link.download = `beat_snapshot_${new Date().toISOString().replace(/[:.]/g, '-')}.${format}`;
    link.href = dataUrl;
    link.click();
}

window.startSimulation = startSimulation;
window.stopSimulation = stopSimulation;
window.resetSimulation = resetSimulation;
window.setSpeed = setSpeed;
window.scrollHistory = scrollHistory;
window.goToLive = goToLive;
window.navigateToTime = navigateToTime;
window.exportECG = exportECG;
window.exportBeatSnapshot = exportBeatSnapshot;

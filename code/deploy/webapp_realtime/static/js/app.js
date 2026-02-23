// ---------------------------------------------------------------------------
// ECG Real-Input Classification - Frontend
// ---------------------------------------------------------------------------

var ecgCanvas = document.getElementById('ecgCanvas');
var ecgCtx = ecgCanvas.getContext('2d');
var beatCanvas = document.getElementById('beatCanvas');
var beatCtx = beatCanvas.getContext('2d');

var signalData = [];
var filteredData = [];
var rPeaks = [];
var beats = [];
var samplingRate = 360;
var totalSamples = 0;

var viewStartSec = 0;
var viewWindowSec = 5;

// Drag state
var isDragging = false;
var lastDragX = 0;

// Selected beat
var selectedBeatIndex = -1;

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
function init() {
    fetchModelInfo();
    setupUpload();
    resizeCanvases();
    window.addEventListener('resize', resizeCanvases);
}

function resizeCanvases() {
    var rect = ecgCanvas.getBoundingClientRect();
    var dpr = window.devicePixelRatio || 1;
    ecgCanvas.width = rect.width * dpr;
    ecgCanvas.height = rect.height * dpr;
    ecgCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

    var beatRect = beatCanvas.getBoundingClientRect();
    beatCanvas.width = beatRect.width * dpr;
    beatCanvas.height = beatRect.height * dpr;
    beatCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

    drawECG();
    if (selectedBeatIndex >= 0 && selectedBeatIndex < beats.length) {
        drawBeatWaveform(beats[selectedBeatIndex]);
    }
}

// ---------------------------------------------------------------------------
// Model info
// ---------------------------------------------------------------------------
function fetchModelInfo() {
    fetch('/api/model_info')
        .then(function(r) { return r.json(); })
        .then(function(data) {
            if (data.name) {
                document.getElementById('modelName').textContent = data.name;
            }
        })
        .catch(function() {});
}

// ---------------------------------------------------------------------------
// File upload
// ---------------------------------------------------------------------------
function setupUpload() {
    var uploadArea = document.getElementById('uploadArea');
    var fileInput = document.getElementById('fileInput');

    uploadArea.addEventListener('click', function() {
        fileInput.click();
    });

    fileInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            uploadFile(e.target.files[0]);
        }
    });

    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    uploadArea.addEventListener('dragleave', function() {
        uploadArea.classList.remove('dragover');
    });
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            uploadFile(e.dataTransfer.files[0]);
        }
    });

    // Drag on ECG canvas
    ecgCanvas.style.cursor = 'grab';
    ecgCanvas.addEventListener('mousedown', function(e) { startDrag(e.clientX); });
    ecgCanvas.addEventListener('mousemove', function(e) { doDrag(e.clientX); });
    ecgCanvas.addEventListener('mouseup', endDrag);
    ecgCanvas.addEventListener('mouseleave', endDrag);

    ecgCanvas.addEventListener('touchstart', function(e) {
        e.preventDefault();
        startDrag(e.touches[0].clientX);
    });
    ecgCanvas.addEventListener('touchmove', function(e) {
        e.preventDefault();
        doDrag(e.touches[0].clientX);
    });
    ecgCanvas.addEventListener('touchend', endDrag);

    // Click on ECG canvas to select beat
    ecgCanvas.addEventListener('click', function(e) {
        if (isDragging) return;
        handleCanvasClick(e);
    });
}

function uploadFile(file) {
    var formData = new FormData();
    formData.append('file', file);
    formData.append('sampling_rate', document.getElementById('samplingRate').value);
    formData.append('ecg_column', document.getElementById('ecgColumn').value);

    document.getElementById('uploadPrompt').style.display = 'none';
    document.getElementById('uploadStatus').style.display = 'block';
    document.getElementById('uploadFilename').textContent = file.name;
    document.getElementById('uploadInfo').textContent = 'Uploading...';
    document.getElementById('processBtn').disabled = true;

    fetch('/api/upload', {
        method: 'POST',
        body: formData
    })
    .then(function(r) { return r.json(); })
    .then(function(data) {
        if (data.error) {
            document.getElementById('uploadInfo').textContent = 'Error: ' + data.error;
            return;
        }
        totalSamples = data.samples;
        samplingRate = data.sampling_rate;
        document.getElementById('uploadInfo').textContent =
            data.samples + ' samples | ' + data.sampling_rate + ' Hz | ' +
            data.duration_sec + 's | Column: ' + data.ecg_column;
        document.getElementById('processBtn').disabled = false;

        // Load signal for display
        loadSignal();
    })
    .catch(function(err) {
        document.getElementById('uploadInfo').textContent = 'Upload failed: ' + err;
    });
}

function loadSignal() {
    fetch('/api/stream?start=0&end=' + totalSamples)
        .then(function(r) { return r.json(); })
        .then(function(data) {
            signalData = data.signal;
            filteredData = data.filtered || [];
            samplingRate = data.sampling_rate;
            totalSamples = data.total_samples;
            viewStartSec = 0;

            document.getElementById('ecgSection').style.display = 'block';
            resizeCanvases();
            drawECG();
        })
        .catch(function() {});
}

// ---------------------------------------------------------------------------
// Processing
// ---------------------------------------------------------------------------
function processECG() {
    var btn = document.getElementById('processBtn');
    btn.disabled = true;
    btn.textContent = 'Processing...';

    document.getElementById('processingIndicator').style.display = 'block';

    var modelVersion = document.getElementById('modelSelect').value;

    // Change model if needed
    fetch('/api/change_model', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({model_version: modelVersion})
    })
    .then(function() {
        return fetch('/api/process', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({model_version: modelVersion})
        });
    })
    .then(function(r) { return r.json(); })
    .then(function(data) {
        document.getElementById('processingIndicator').style.display = 'none';
        btn.disabled = false;
        btn.textContent = 'Process ECG';

        if (data.error) {
            alert('Processing error: ' + data.error);
            return;
        }

        rPeaks = data.r_peaks || [];
        beats = data.beats || [];

        fetchModelInfo();
        updateStats(data);
        renderClassificationList(data.beats);

        // Reload signal (now with filtered data)
        loadSignal();

        document.getElementById('statsBar').style.display = 'flex';
        document.getElementById('beatSection').style.display = 'block';
        document.getElementById('resultsSection').style.display = 'grid';

        if (beats.length > 0) {
            selectBeat(0);
        }
    })
    .catch(function(err) {
        document.getElementById('processingIndicator').style.display = 'none';
        btn.disabled = false;
        btn.textContent = 'Process ECG';
        alert('Error: ' + err);
    });
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------
function updateStats(data) {
    document.getElementById('totalBeats').textContent = data.total_beats;
    document.getElementById('normalBeats').textContent = data.normal_count;
    document.getElementById('abnormalBeats').textContent = data.abnormal_count;

    var totalClassified = data.normal_count + data.abnormal_count;
    if (totalClassified > 0) {
        var pct = ((data.normal_count / totalClassified) * 100).toFixed(1);
        document.getElementById('normalPct').textContent = pct + '%';
    }

    var durationSec = totalSamples / samplingRate;
    document.getElementById('duration').textContent = durationSec.toFixed(1);

    if (data.total_beats > 1 && durationSec > 0) {
        var bpm = Math.round((data.total_beats / durationSec) * 60);
        document.getElementById('avgHR').textContent = bpm;
    }

    // Summary bar
    if (totalClassified > 0) {
        var normalPct = (data.normal_count / totalClassified) * 100;
        document.getElementById('summaryBar').style.width = normalPct + '%';
        if (normalPct > 80) {
            document.getElementById('summaryBar').style.background = '#00ff88';
        } else if (normalPct > 50) {
            document.getElementById('summaryBar').style.background = '#ffd700';
        } else {
            document.getElementById('summaryBar').style.background = '#ff4757';
        }
        document.getElementById('summaryText').textContent =
            data.normal_count + ' normal / ' + data.abnormal_count + ' abnormal out of ' +
            totalClassified + ' classified beats (' + data.total_beats + ' total detected)';
    }
}

// ---------------------------------------------------------------------------
// Classification list
// ---------------------------------------------------------------------------
function renderClassificationList(beatList) {
    var list = document.getElementById('classificationList');
    list.innerHTML = '';

    if (!beatList || beatList.length === 0) {
        list.innerHTML = '<p style="color: #888; text-align: center;">No beats detected.</p>';
        return;
    }

    for (var i = 0; i < beatList.length; i++) {
        var b = beatList[i];
        var cls = 'normal';
        if (b.predicted === 'ABNORMAL') cls = 'abnormal';
        else if (b.predicted === 'WAITING') cls = 'waiting';

        var item = document.createElement('div');
        item.className = 'classification-item ' + cls;
        item.setAttribute('data-beat-index', i);
        item.onclick = (function(idx) {
            return function() { selectBeat(idx); };
        })(i);

        var info = document.createElement('div');
        info.className = 'beat-info';
        info.innerHTML = '<div>Beat #' + b.beat_index + '</div>' +
            '<div class="beat-time">' + formatTime(b.time_sec) +
            ' | P(abn)=' + b.probability.toFixed(4) + '</div>';

        var badge = document.createElement('span');
        badge.className = 'prediction-badge ' + cls;
        badge.textContent = b.predicted;

        item.appendChild(info);
        item.appendChild(badge);
        list.appendChild(item);
    }
}

function formatTime(sec) {
    var m = Math.floor(sec / 60);
    var s = sec - m * 60;
    return m + ':' + (s < 10 ? '0' : '') + s.toFixed(3);
}

// ---------------------------------------------------------------------------
// Beat selection
// ---------------------------------------------------------------------------
function selectBeat(index) {
    if (index < 0 || index >= beats.length) return;
    selectedBeatIndex = index;
    var b = beats[index];

    document.getElementById('beatIndexDisplay').textContent = '#' + b.beat_index;
    document.getElementById('beatTimeDisplay').textContent = formatTime(b.time_sec);
    document.getElementById('predictionDisplay').textContent = b.predicted;
    document.getElementById('predictionDisplay').style.color =
        b.predicted === 'NORMAL' ? '#00ff88' :
        b.predicted === 'ABNORMAL' ? '#ff4757' : '#ffd700';
    document.getElementById('probDisplay').textContent = b.probability.toFixed(4);

    drawBeatWaveform(b);

    // Scroll ECG view to show this beat
    var beatTimeSec = b.time_sec;
    viewStartSec = Math.max(0, beatTimeSec - viewWindowSec / 2);
    drawECG();
    updateTimeRange();

    // Highlight in list
    var items = document.querySelectorAll('.classification-item');
    for (var i = 0; i < items.length; i++) {
        items[i].style.outline = '';
    }
    var selected = document.querySelector('[data-beat-index="' + index + '"]');
    if (selected) {
        selected.style.outline = '2px solid #00ff88';
        selected.scrollIntoView({block: 'nearest'});
    }
}

// ---------------------------------------------------------------------------
// ECG canvas drawing
// ---------------------------------------------------------------------------
function drawECG() {
    var rect = ecgCanvas.getBoundingClientRect();
    var w = rect.width;
    var h = rect.height;

    ecgCtx.clearRect(0, 0, w, h);

    // Background
    ecgCtx.fillStyle = '#0a0a1a';
    ecgCtx.fillRect(0, 0, w, h);

    if (signalData.length === 0) return;

    // Determine visible sample range
    // signalData may be downsampled; compute effective rate
    var effectiveRate = signalData.length / (totalSamples / samplingRate);
    var startSample = Math.floor(viewStartSec * effectiveRate);
    var endSample = Math.floor((viewStartSec + viewWindowSec) * effectiveRate);
    startSample = Math.max(0, startSample);
    endSample = Math.min(signalData.length, endSample);

    var visibleSig = signalData.slice(startSample, endSample);

    if (visibleSig.length === 0) return;

    // Grid lines
    ecgCtx.strokeStyle = 'rgba(0, 255, 136, 0.07)';
    ecgCtx.lineWidth = 0.5;
    for (var gx = 0; gx < w; gx += 30) {
        ecgCtx.beginPath();
        ecgCtx.moveTo(gx, 0);
        ecgCtx.lineTo(gx, h);
        ecgCtx.stroke();
    }
    for (var gy = 0; gy < h; gy += 30) {
        ecgCtx.beginPath();
        ecgCtx.moveTo(0, gy);
        ecgCtx.lineTo(w, gy);
        ecgCtx.stroke();
    }

    // Compute Y range
    var minVal = Infinity, maxVal = -Infinity;
    for (var i = 0; i < visibleSig.length; i++) {
        if (visibleSig[i] < minVal) minVal = visibleSig[i];
        if (visibleSig[i] > maxVal) maxVal = visibleSig[i];
    }
    var range = maxVal - minVal || 1;
    var padding = range * 0.1;
    minVal -= padding;
    maxVal += padding;
    range = maxVal - minVal;

    // Draw signal
    ecgCtx.strokeStyle = '#00ff88';
    ecgCtx.lineWidth = 1.5;
    ecgCtx.beginPath();
    for (var i = 0; i < visibleSig.length; i++) {
        var x = (i / visibleSig.length) * w;
        var y = h - ((visibleSig[i] - minVal) / range) * h;
        if (i === 0) ecgCtx.moveTo(x, y);
        else ecgCtx.lineTo(x, y);
    }
    ecgCtx.stroke();

    // Draw R-peak markers
    if (rPeaks.length > 0) {
        var viewStartSample = Math.floor(viewStartSec * samplingRate);
        var viewEndSample = Math.floor((viewStartSec + viewWindowSec) * samplingRate);

        for (var p = 0; p < rPeaks.length; p++) {
            var rSample = rPeaks[p];
            if (rSample < viewStartSample || rSample > viewEndSample) continue;

            var xPos = ((rSample - viewStartSample) / (viewEndSample - viewStartSample)) * w;

            // Find the beat classification for this R-peak
            var beatInfo = null;
            for (var bi = 0; bi < beats.length; bi++) {
                if (beats[bi].r_peak === rSample) {
                    beatInfo = beats[bi];
                    break;
                }
            }

            var markerColor = '#ffd700';
            if (beatInfo) {
                if (beatInfo.predicted === 'NORMAL') markerColor = '#00ff88';
                else if (beatInfo.predicted === 'ABNORMAL') markerColor = '#ff4757';
            }

            // Vertical line
            ecgCtx.strokeStyle = markerColor;
            ecgCtx.lineWidth = 1;
            ecgCtx.globalAlpha = 0.5;
            ecgCtx.beginPath();
            ecgCtx.moveTo(xPos, 0);
            ecgCtx.lineTo(xPos, h);
            ecgCtx.stroke();
            ecgCtx.globalAlpha = 1.0;

            // Triangle marker at top
            ecgCtx.fillStyle = markerColor;
            ecgCtx.beginPath();
            ecgCtx.moveTo(xPos - 5, 0);
            ecgCtx.lineTo(xPos + 5, 0);
            ecgCtx.lineTo(xPos, 10);
            ecgCtx.closePath();
            ecgCtx.fill();
        }

        // Highlight selected beat
        if (selectedBeatIndex >= 0 && selectedBeatIndex < beats.length) {
            var sb = beats[selectedBeatIndex];
            if (sb.r_peak >= viewStartSample && sb.r_peak <= viewEndSample) {
                var sx = ((sb.r_peak - viewStartSample) / (viewEndSample - viewStartSample)) * w;
                ecgCtx.strokeStyle = '#ffffff';
                ecgCtx.lineWidth = 2;
                ecgCtx.setLineDash([4, 4]);
                ecgCtx.beginPath();
                ecgCtx.moveTo(sx, 0);
                ecgCtx.lineTo(sx, h);
                ecgCtx.stroke();
                ecgCtx.setLineDash([]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Beat waveform drawing
// ---------------------------------------------------------------------------
function drawBeatWaveform(beat) {
    var waveform = beat.beat_waveform;
    if (!waveform || waveform.length === 0) return;

    var rect = beatCanvas.getBoundingClientRect();
    var w = rect.width;
    var h = rect.height;

    beatCtx.clearRect(0, 0, w, h);
    beatCtx.fillStyle = '#0a0a1a';
    beatCtx.fillRect(0, 0, w, h);

    // Grid
    beatCtx.strokeStyle = 'rgba(0, 255, 136, 0.07)';
    beatCtx.lineWidth = 0.5;
    for (var gx = 0; gx < w; gx += 20) {
        beatCtx.beginPath();
        beatCtx.moveTo(gx, 0);
        beatCtx.lineTo(gx, h);
        beatCtx.stroke();
    }
    for (var gy = 0; gy < h; gy += 20) {
        beatCtx.beginPath();
        beatCtx.moveTo(0, gy);
        beatCtx.lineTo(w, gy);
        beatCtx.stroke();
    }

    var minVal = Infinity, maxVal = -Infinity;
    for (var i = 0; i < waveform.length; i++) {
        if (waveform[i] < minVal) minVal = waveform[i];
        if (waveform[i] > maxVal) maxVal = waveform[i];
    }
    var range = maxVal - minVal || 1;
    var padding = range * 0.15;
    minVal -= padding;
    maxVal += padding;
    range = maxVal - minVal;

    var color = beat.predicted === 'ABNORMAL' ? '#ff4757' :
                beat.predicted === 'NORMAL' ? '#00ff88' : '#ffd700';

    beatCtx.strokeStyle = color;
    beatCtx.lineWidth = 2;
    beatCtx.beginPath();
    for (var i = 0; i < waveform.length; i++) {
        var x = (i / waveform.length) * w;
        var y = h - ((waveform[i] - minVal) / range) * h;
        if (i === 0) beatCtx.moveTo(x, y);
        else beatCtx.lineTo(x, y);
    }
    beatCtx.stroke();

    // R-peak marker
    if (beat.r_peak_pos_in_beat !== undefined) {
        var rX = (beat.r_peak_pos_in_beat / waveform.length) * w;
        beatCtx.strokeStyle = '#ffffff';
        beatCtx.lineWidth = 1;
        beatCtx.setLineDash([3, 3]);
        beatCtx.beginPath();
        beatCtx.moveTo(rX, 0);
        beatCtx.lineTo(rX, h);
        beatCtx.stroke();
        beatCtx.setLineDash([]);
    }
}

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------
function scrollView(deltaSec) {
    var maxSec = (totalSamples / samplingRate) - viewWindowSec;
    viewStartSec = Math.max(0, Math.min(maxSec, viewStartSec + deltaSec));
    drawECG();
    updateTimeRange();
}

function updateTimeRange() {
    var endSec = viewStartSec + viewWindowSec;
    document.getElementById('currentTimeRange').textContent =
        formatTime(viewStartSec) + ' - ' + formatTime(endSec);
}

// Drag to scroll
function startDrag(x) {
    isDragging = true;
    lastDragX = x;
    ecgCanvas.style.cursor = 'grabbing';
}

function doDrag(x) {
    if (!isDragging) return;
    var deltaX = x - lastDragX;
    lastDragX = x;
    var canvasWidth = ecgCanvas.getBoundingClientRect().width;
    var secondsPerPixel = viewWindowSec / canvasWidth;
    var deltaSec = -deltaX * secondsPerPixel;
    if (Math.abs(deltaSec) > 0.005) {
        scrollView(deltaSec);
    }
}

function endDrag() {
    isDragging = false;
    ecgCanvas.style.cursor = 'grab';
}

// Click on canvas to find nearest R-peak
function handleCanvasClick(e) {
    if (rPeaks.length === 0 || beats.length === 0) return;

    var rect = ecgCanvas.getBoundingClientRect();
    var clickX = e.clientX - rect.left;
    var frac = clickX / rect.width;

    var viewStartSample = Math.floor(viewStartSec * samplingRate);
    var viewEndSample = Math.floor((viewStartSec + viewWindowSec) * samplingRate);
    var clickSample = viewStartSample + frac * (viewEndSample - viewStartSample);

    // Find nearest R-peak
    var nearestDist = Infinity;
    var nearestBeatIdx = -1;
    for (var i = 0; i < beats.length; i++) {
        var dist = Math.abs(beats[i].r_peak - clickSample);
        if (dist < nearestDist) {
            nearestDist = dist;
            nearestBeatIdx = i;
        }
    }

    // Only select if close enough (within 0.5 seconds)
    if (nearestBeatIdx >= 0 && nearestDist < samplingRate * 0.5) {
        selectBeat(nearestBeatIdx);
    }
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------
function exportECG(format) {
    format = format || 'png';

    if (signalData.length === 0) {
        alert('No ECG data loaded.');
        return;
    }

    var EXPORT_WIDTH = 4000;
    var ROW_HEIGHT = 250;
    var HEADER_HEIGHT = 80;
    var FOOTER_HEIGHT = 60;
    var SECONDS_PER_ROW = 10;
    var GRID_SPACING = 30;

    var totalSec = totalSamples / samplingRate;
    var numRows = Math.ceil(totalSec / SECONDS_PER_ROW);
    var exportHeight = HEADER_HEIGHT + numRows * ROW_HEIGHT + FOOTER_HEIGHT;

    var exportCanvas = document.createElement('canvas');
    exportCanvas.width = EXPORT_WIDTH;
    exportCanvas.height = Math.min(exportHeight, 10000);
    var ectx = exportCanvas.getContext('2d');

    // Background
    ectx.fillStyle = '#0a0a1a';
    ectx.fillRect(0, 0, EXPORT_WIDTH, exportCanvas.height);

    // Header
    ectx.fillStyle = '#00ff88';
    ectx.font = 'bold 28px Courier New';
    ectx.fillText('ECG Recording - ' + (document.getElementById('uploadFilename').textContent || 'Unknown'), 20, 35);
    ectx.fillStyle = '#888';
    ectx.font = '18px Courier New';
    ectx.fillText('Model: ' + document.getElementById('modelName').textContent +
        ' | Rate: ' + samplingRate + ' Hz | Duration: ' + totalSec.toFixed(1) + 's', 20, 60);

    // Draw rows
    var effectiveRate = signalData.length / totalSec;
    var globalMin = Infinity, globalMax = -Infinity;
    for (var i = 0; i < signalData.length; i++) {
        if (signalData[i] < globalMin) globalMin = signalData[i];
        if (signalData[i] > globalMax) globalMax = signalData[i];
    }
    var globalRange = globalMax - globalMin || 1;

    var maxRows = Math.floor((exportCanvas.height - HEADER_HEIGHT - FOOTER_HEIGHT) / ROW_HEIGHT);
    numRows = Math.min(numRows, maxRows);

    for (var row = 0; row < numRows; row++) {
        var rowY = HEADER_HEIGHT + row * ROW_HEIGHT;
        var rowStartSec = row * SECONDS_PER_ROW;
        var rowEndSec = Math.min((row + 1) * SECONDS_PER_ROW, totalSec);
        var sStart = Math.floor(rowStartSec * effectiveRate);
        var sEnd = Math.floor(rowEndSec * effectiveRate);

        // Grid
        ectx.strokeStyle = 'rgba(0, 255, 136, 0.1)';
        ectx.lineWidth = 0.5;
        for (var gx = 0; gx < EXPORT_WIDTH; gx += GRID_SPACING) {
            ectx.beginPath();
            ectx.moveTo(gx, rowY);
            ectx.lineTo(gx, rowY + ROW_HEIGHT);
            ectx.stroke();
        }
        for (var gy = rowY; gy < rowY + ROW_HEIGHT; gy += GRID_SPACING) {
            ectx.beginPath();
            ectx.moveTo(0, gy);
            ectx.lineTo(EXPORT_WIDTH, gy);
            ectx.stroke();
        }

        // Time label
        ectx.fillStyle = '#888';
        ectx.font = '12px Courier New';
        ectx.fillText(formatTime(rowStartSec), 5, rowY + 15);

        // Signal
        var seg = signalData.slice(sStart, sEnd);
        if (seg.length > 0) {
            ectx.strokeStyle = '#00ff88';
            ectx.lineWidth = 1;
            ectx.beginPath();
            for (var j = 0; j < seg.length; j++) {
                var x = (j / seg.length) * EXPORT_WIDTH;
                var y = rowY + ROW_HEIGHT - ((seg[j] - globalMin) / globalRange) * (ROW_HEIGHT - 20) - 10;
                if (j === 0) ectx.moveTo(x, y);
                else ectx.lineTo(x, y);
            }
            ectx.stroke();
        }

        // R-peak markers in this row
        var rStart = Math.floor(rowStartSec * samplingRate);
        var rEnd = Math.floor(rowEndSec * samplingRate);
        for (var p = 0; p < rPeaks.length; p++) {
            if (rPeaks[p] >= rStart && rPeaks[p] < rEnd) {
                var rx = ((rPeaks[p] - rStart) / (rEnd - rStart)) * EXPORT_WIDTH;
                var beatInfo = null;
                for (var bi = 0; bi < beats.length; bi++) {
                    if (beats[bi].r_peak === rPeaks[p]) {
                        beatInfo = beats[bi];
                        break;
                    }
                }
                var mc = '#ffd700';
                if (beatInfo && beatInfo.predicted === 'NORMAL') mc = '#00ff88';
                else if (beatInfo && beatInfo.predicted === 'ABNORMAL') mc = '#ff4757';

                ectx.strokeStyle = mc;
                ectx.lineWidth = 1;
                ectx.globalAlpha = 0.4;
                ectx.beginPath();
                ectx.moveTo(rx, rowY);
                ectx.lineTo(rx, rowY + ROW_HEIGHT);
                ectx.stroke();
                ectx.globalAlpha = 1.0;

                ectx.fillStyle = mc;
                ectx.beginPath();
                ectx.moveTo(rx - 4, rowY);
                ectx.lineTo(rx + 4, rowY);
                ectx.lineTo(rx, rowY + 8);
                ectx.closePath();
                ectx.fill();
            }
        }

        // Row separator
        ectx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        ectx.lineWidth = 1;
        ectx.beginPath();
        ectx.moveTo(0, rowY + ROW_HEIGHT);
        ectx.lineTo(EXPORT_WIDTH, rowY + ROW_HEIGHT);
        ectx.stroke();
    }

    // Footer
    var footerY = HEADER_HEIGHT + numRows * ROW_HEIGHT + 20;
    ectx.fillStyle = '#888';
    ectx.font = '14px Courier New';
    var normal = 0, abnormal = 0;
    for (var i = 0; i < beats.length; i++) {
        if (beats[i].predicted === 'NORMAL') normal++;
        else if (beats[i].predicted === 'ABNORMAL') abnormal++;
    }
    ectx.fillText('Total beats: ' + beats.length + ' | Normal: ' + normal +
        ' | Abnormal: ' + abnormal + ' | Generated: ' + new Date().toISOString(), 20, footerY);

    // Download
    var mimeType = format === 'jpeg' ? 'image/jpeg' : 'image/png';
    var quality = format === 'jpeg' ? 0.95 : undefined;
    var dataUrl = exportCanvas.toDataURL(mimeType, quality);
    var link = document.createElement('a');
    link.download = 'ecg_export.' + format;
    link.href = dataUrl;
    link.click();
}

// ---------------------------------------------------------------------------
// Initialize
// ---------------------------------------------------------------------------
init();

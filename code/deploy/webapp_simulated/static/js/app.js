const canvas = document.getElementById('ecgCanvas');
const ctx = canvas.getContext('2d');

// Beat snapshot canvas
const beatCanvas = document.getElementById('beatCanvas');
const beatCtx = beatCanvas.getContext('2d');

let ecgData = [];
let annotations = [];
let currentIndex = 0;
let isRunning = false;
let animationId = null;
let displayBuffer = [];
let classifications = [];
let falseDetections = [];
let beatTimes = [];
let speedMultiplier = 1;
let currentBeatWaveform = null;
let currentRPeakPos = 70;
let currentBeatLength = 188;

// Graph height tracking
let maxGraphHeight = 300;
const MIN_GRAPH_HEIGHT = 300;
const MAX_GRAPH_HEIGHT = 800;

// Y-axis scale tracking
let globalMinVal = Infinity;
let globalMaxVal = -Infinity;

// History navigation
let viewOffset = 0;
let isLive = true;

// High-speed stability
let isClassifying = false;
let classificationQueue = [];
let processedBeats = new Set();
const MAX_CLASSIFICATIONS = 1000;
const MAX_FALSE_DETECTIONS = 100;

const SAMPLING_RATE = 360;
const DISPLAY_SECONDS = 5;
const DISPLAY_SAMPLES = SAMPLING_RATE * DISPLAY_SECONDS;

// ============================================================
// AUTO-BATCH EXPORT SYSTEM
// ============================================================
const AUTO_BATCH_INTERVAL_SECONDS = 120;
const AUTO_BATCH_INTERVAL_SAMPLES = AUTO_BATCH_INTERVAL_SECONDS * SAMPLING_RATE;
const MIN_BATCH_SECONDS = 5;
const MIN_BATCH_SAMPLES = MIN_BATCH_SECONDS * SAMPLING_RATE;
const BATCH_CHECK_INTERVAL_MS = 5000;
const BATCH_GRID_SPACING = 30;
let savedBatches = [];
let lastBatchEndSample = 0;
let autoBatchEnabled = true;

// Speed control
function setSpeed(speed) {
    speedMultiplier = speed;
    document.getElementById('speedValue').textContent = speed + 'x';
    document.querySelectorAll('.speed-btn').forEach(function(btn) {
        btn.classList.remove('active');
        if (btn.textContent === speed + 'x') btn.classList.add('active');
    });
}

// History navigation functions
function scrollHistory(seconds) {
    if (currentIndex < DISPLAY_SAMPLES) return;

    viewOffset += seconds;
    var maxHistory = -currentIndex / SAMPLING_RATE;
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
    var targetOffset = (sampleIndex - currentIndex + DISPLAY_SAMPLES / 2) / SAMPLING_RATE;
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
    var indicator = document.getElementById('historyIndicator');
    var fwdBtn = document.getElementById('fwdBtn');
    var fwd5Btn = document.getElementById('fwd5Btn');

    if (isLive) {
        indicator.style.display = 'none';
        fwdBtn.disabled = true;
        fwd5Btn.disabled = true;
    } else {
        indicator.style.display = 'inline';
        fwdBtn.disabled = false;
        fwd5Btn.disabled = false;
    }
}

// Update graph height dynamically
function updateGraphHeight(requestedHeight) {
    var newHeight = Math.max(MIN_GRAPH_HEIGHT, Math.min(MAX_GRAPH_HEIGHT, requestedHeight));
    if (newHeight > maxGraphHeight) {
        maxGraphHeight = newHeight;
        canvas.style.height = maxGraphHeight + 'px';
        resizeCanvas();
    }
}

// Resize canvas to be pixel-perfect
function resizeCanvas() {
    canvas.style.height = maxGraphHeight + 'px';

    var rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    var heightToUse = Math.max(rect.height, maxGraphHeight);
    canvas.height = heightToUse * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    // Also resize beat canvas
    var beatRect = beatCanvas.getBoundingClientRect();
    beatCanvas.width = beatRect.width * window.devicePixelRatio;
    beatCanvas.height = beatRect.height * window.devicePixelRatio;
    beatCtx.scale(window.devicePixelRatio, window.devicePixelRatio);

    if (currentBeatWaveform) {
        drawBeatWaveform(currentBeatWaveform);
    }
}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

// ============================================================
// DRAG INTERACTION FOR SCROLLABLE HISTORY
// ============================================================
let isDragging = false;
let lastDragX = 0;

canvas.style.cursor = 'grab';

function startDrag(x) {
    isDragging = true;
    lastDragX = x;
    canvas.style.cursor = 'grabbing';
}

function drag(x) {
    if (!isDragging) return;

    var deltaX = x - lastDragX;
    lastDragX = x;

    var canvasWidth = canvas.getBoundingClientRect().width;
    var secondsPerPixel = DISPLAY_SECONDS / canvasWidth;
    var deltaSeconds = -deltaX * secondsPerPixel;

    if (Math.abs(deltaSeconds) > 0.01) {
        scrollHistory(deltaSeconds);
    }
}

function endDrag() {
    isDragging = false;
    canvas.style.cursor = 'grab';
}

// Mouse events
canvas.addEventListener('mousedown', function(e) { startDrag(e.clientX); });
canvas.addEventListener('mousemove', function(e) { drag(e.clientX); });
canvas.addEventListener('mouseup', function() { endDrag(); });
canvas.addEventListener('mouseleave', function() { endDrag(); });

// Touch events for mobile
canvas.addEventListener('touchstart', function(e) {
    e.preventDefault();
    startDrag(e.touches[0].clientX);
});
canvas.addEventListener('touchmove', function(e) {
    e.preventDefault();
    drag(e.touches[0].clientX);
});
canvas.addEventListener('touchend', function() { endDrag(); });

// ============================================================
// EXPORT TO MEDICAL IMAGE
// ============================================================
function exportECG(format) {
    format = format || 'png';
    var startSample = 0;
    var endSample = currentIndex > 0 ? currentIndex : Math.min(DISPLAY_SAMPLES, ecgData.length);

    var EXPORT_MAX_WIDTH = 10000;
    var EXPORT_MAX_HEIGHT = 10000;
    var ROW_HEIGHT = 250;
    var HEADER_HEIGHT = 90;
    var FOOTER_HEIGHT = 80;
    var SECONDS_PER_ROW = 30;
    var PIXELS_PER_SECOND = EXPORT_MAX_WIDTH / SECONDS_PER_ROW;

    var totalSeconds = endSample / SAMPLING_RATE;
    var totalSamples = endSample - startSample;
    var samplesPerRow = Math.round(SECONDS_PER_ROW * SAMPLING_RATE);
    var numRows = Math.ceil(totalSamples / samplesPerRow);

    var maxRowsPerPart = Math.floor((EXPORT_MAX_HEIGHT - HEADER_HEIGHT - FOOTER_HEIGHT) / ROW_HEIGHT);
    var numParts = Math.ceil(numRows / maxRowsPerPart);

    var fullBuffer = [];
    for (var i = startSample; i < endSample && i < ecgData.length; i++) {
        fullBuffer.push(ecgData[i]);
    }

    var expGlobalMinVal = Math.min.apply(null, fullBuffer);
    var expGlobalMaxVal = Math.max.apply(null, fullBuffer);
    var expGlobalRange = expGlobalMaxVal - expGlobalMinVal || 1;

    var modelName = document.getElementById('modelName').textContent;
    var timestamp = new Date().toISOString();

    console.log('[ECG] Exporting ' + totalSeconds.toFixed(2) + 's recording: ' + numRows + ' rows across ' + numParts + ' part(s)');

    for (var partIdx = 0; partIdx < numParts; partIdx++) {
        var rowsInThisPart = Math.min(maxRowsPerPart, numRows - partIdx * maxRowsPerPart);
        var exportWidth = EXPORT_MAX_WIDTH;
        var exportHeight = HEADER_HEIGHT + rowsInThisPart * ROW_HEIGHT + FOOTER_HEIGHT;

        var exportCanvas = document.createElement('canvas');
        exportCanvas.width = exportWidth;
        exportCanvas.height = exportHeight;
        var exportCtx = exportCanvas.getContext('2d');

        exportCtx.fillStyle = '#ffffff';
        exportCtx.fillRect(0, 0, exportWidth, exportHeight);

        exportCtx.fillStyle = '#333333';
        exportCtx.font = 'bold 18px Arial';
        var partLabel = numParts > 1 ? ' (Part ' + (partIdx + 1) + ' of ' + numParts + ')' : '';
        exportCtx.fillText('ECG Analysis Report - Complete Recording' + partLabel, 20, 30);

        exportCtx.font = '12px Arial';
        exportCtx.fillStyle = '#666666';
        exportCtx.fillText('Model: ' + modelName, 20, 50);
        exportCtx.fillText('Timestamp: ' + timestamp, 20, 68);

        var partStartRow = partIdx * maxRowsPerPart;
        var partEndRow = partStartRow + rowsInThisPart;
        var partStartSample = partStartRow * samplesPerRow;
        var partEndSample = Math.min(partEndRow * samplesPerRow, totalSamples);
        var partStartTime = (partStartSample / SAMPLING_RATE).toFixed(2);
        var partEndTime = (partEndSample / SAMPLING_RATE).toFixed(2);

        exportCtx.fillText('Time Range: ' + partStartTime + 's - ' + partEndTime + 's | Total: ' + totalSeconds.toFixed(2) + 's', 300, 50);
        exportCtx.fillText('Rows ' + (partStartRow + 1) + '-' + partEndRow + ' of ' + numRows + ' | ' + SECONDS_PER_ROW + 's per row', 300, 68);

        for (var rowInPart = 0; rowInPart < rowsInThisPart; rowInPart++) {
            var globalRowIdx = partStartRow + rowInPart;
            var rowStartSample = globalRowIdx * samplesPerRow;
            var rowEndSample = Math.min(rowStartSample + samplesPerRow, totalSamples);

            if (rowStartSample >= totalSamples) break;

            var rowBuffer = fullBuffer.slice(rowStartSample, rowEndSample);
            if (rowBuffer.length === 0) continue;

            var graphX = 100;
            var graphY = HEADER_HEIGHT + rowInPart * ROW_HEIGHT + 30;
            var graphWidth = exportWidth - 120;
            var graphHeight = ROW_HEIGHT - 50;

            var rowStartTime = (rowStartSample / SAMPLING_RATE);
            var rowEndTime = (rowEndSample / SAMPLING_RATE);

            var rowDataWidth = (rowBuffer.length / samplesPerRow) * graphWidth;
            var isPartialRow = rowBuffer.length < samplesPerRow;

            exportCtx.fillStyle = '#1a5276';
            exportCtx.font = 'bold 14px Arial';
            exportCtx.fillText(formatTime(rowStartTime), 10, graphY + graphHeight / 2 + 5);

            if (isPartialRow) {
                var endLabelX = graphX + rowDataWidth + 10;
                exportCtx.fillText(formatTime(rowEndTime), endLabelX, graphY + graphHeight / 2 + 5);

                exportCtx.strokeStyle = '#aaaaaa';
                exportCtx.lineWidth = 2;
                exportCtx.setLineDash([5, 5]);
                exportCtx.beginPath();
                exportCtx.moveTo(graphX + rowDataWidth, graphY);
                exportCtx.lineTo(graphX + rowDataWidth, graphY + graphHeight);
                exportCtx.stroke();
                exportCtx.setLineDash([]);

                exportCtx.fillStyle = '#888888';
                exportCtx.font = 'italic 10px Arial';
                exportCtx.fillText('(Recording End)', graphX + rowDataWidth + 10, graphY + graphHeight / 2 + 20);
            } else {
                exportCtx.fillText(formatTime(rowEndTime), exportWidth - 85, graphY + graphHeight / 2 + 5);
            }

            exportCtx.fillStyle = '#7f8c8d';
            exportCtx.font = '10px Arial';
            exportCtx.fillText('Row ' + (globalRowIdx + 1), 10, graphY - 5);

            exportCtx.strokeStyle = '#cccccc';
            exportCtx.lineWidth = 1;
            exportCtx.strokeRect(graphX, graphY, graphWidth, graphHeight);

            // Medical ECG grid (red)
            var gridSpacingSmall = 15;
            var gridSpacingLarge = 75;

            exportCtx.strokeStyle = '#ffcccc';
            exportCtx.lineWidth = 0.5;
            for (var gx = graphX; gx <= graphX + graphWidth; gx += gridSpacingSmall) {
                exportCtx.beginPath();
                exportCtx.moveTo(gx, graphY);
                exportCtx.lineTo(gx, graphY + graphHeight);
                exportCtx.stroke();
            }
            for (var gy = graphY; gy <= graphY + graphHeight; gy += gridSpacingSmall) {
                exportCtx.beginPath();
                exportCtx.moveTo(graphX, gy);
                exportCtx.lineTo(graphX + graphWidth, gy);
                exportCtx.stroke();
            }

            exportCtx.strokeStyle = '#ff9999';
            exportCtx.lineWidth = 1;
            for (var lgx = graphX; lgx <= graphX + graphWidth; lgx += gridSpacingLarge) {
                exportCtx.beginPath();
                exportCtx.moveTo(lgx, graphY);
                exportCtx.lineTo(lgx, graphY + graphHeight);
                exportCtx.stroke();
            }

            // Time markers
            exportCtx.fillStyle = '#666666';
            exportCtx.font = '9px Arial';
            var secondsInRow = (rowEndSample - rowStartSample) / SAMPLING_RATE;
            var actualRowWidth = (rowBuffer.length / samplesPerRow) * graphWidth;
            var timeMarkInterval = SECONDS_PER_ROW > 20 ? 5 : (SECONDS_PER_ROW > 10 ? 2 : 1);

            for (var t = 0; t <= secondsInRow; t += timeMarkInterval) {
                var xPos = graphX + (t / SECONDS_PER_ROW) * graphWidth;
                if (xPos <= graphX + actualRowWidth + 5) {
                    var timeLabel = (rowStartTime + t).toFixed(1) + 's';
                    exportCtx.fillText(timeLabel, xPos - 10, graphY - 3);

                    exportCtx.strokeStyle = '#999999';
                    exportCtx.lineWidth = 1;
                    exportCtx.beginPath();
                    exportCtx.moveTo(xPos, graphY);
                    exportCtx.lineTo(xPos, graphY + 5);
                    exportCtx.stroke();
                }
            }

            // Draw ECG signal
            if (rowBuffer.length >= 2) {
                exportCtx.strokeStyle = '#00aa66';
                exportCtx.lineWidth = 1.5;
                exportCtx.beginPath();

                for (var ri = 0; ri < rowBuffer.length; ri++) {
                    var rx = graphX + (ri / samplesPerRow) * graphWidth;
                    var ry = graphY + graphHeight - ((rowBuffer[ri] - expGlobalMinVal) / expGlobalRange) * (graphHeight - 20) - 10;

                    if (ri === 0) {
                        exportCtx.moveTo(rx, ry);
                    } else {
                        exportCtx.lineTo(rx, ry);
                    }
                }
                exportCtx.stroke();

                // Draw R-peak markers
                annotations.forEach(function(ann) {
                    var globalIdx = ann.sample_index - startSample;
                    if (globalIdx >= rowStartSample && globalIdx < rowEndSample) {
                        var localIdx = globalIdx - rowStartSample;
                        if (localIdx >= 0 && localIdx < rowBuffer.length) {
                            var ax = graphX + (localIdx / samplesPerRow) * graphWidth;
                            var ay = graphY + graphHeight - ((rowBuffer[localIdx] - expGlobalMinVal) / expGlobalRange) * (graphHeight - 20) - 10;

                            var classResult = classifications.find(function(c) { return c.r_peak === ann.sample_index; });
                            if (classResult && classResult.correct === false) {
                                exportCtx.strokeStyle = '#cc8800';
                                exportCtx.lineWidth = 2;
                                exportCtx.beginPath();
                                exportCtx.arc(ax, ay, 6, 0, Math.PI * 2);
                                exportCtx.stroke();
                            }

                            exportCtx.fillStyle = ann.beat_type === 'N' ? '#00aa66' : '#cc3333';
                            exportCtx.beginPath();
                            exportCtx.arc(ax, ay, 3, 0, Math.PI * 2);
                            exportCtx.fill();
                        }
                    }
                });
            }
        }

        // Legend at bottom
        var legendY = exportHeight - 50;
        exportCtx.font = '11px Arial';
        exportCtx.fillStyle = '#00aa66';
        exportCtx.beginPath();
        exportCtx.arc(60, legendY, 5, 0, Math.PI * 2);
        exportCtx.fill();
        exportCtx.fillStyle = '#333333';
        exportCtx.fillText('Normal Beat', 72, legendY + 4);

        exportCtx.fillStyle = '#cc3333';
        exportCtx.beginPath();
        exportCtx.arc(180, legendY, 5, 0, Math.PI * 2);
        exportCtx.fill();
        exportCtx.fillStyle = '#333333';
        exportCtx.fillText('Abnormal Beat', 192, legendY + 4);

        exportCtx.strokeStyle = '#cc8800';
        exportCtx.lineWidth = 2;
        exportCtx.beginPath();
        exportCtx.arc(320, legendY, 7, 0, Math.PI * 2);
        exportCtx.stroke();
        exportCtx.fillStyle = '#333333';
        exportCtx.fillText('False Detection', 335, legendY + 4);

        exportCtx.fillStyle = '#666666';
        exportCtx.font = '10px Arial';
        exportCtx.fillText('Scale: ' + SECONDS_PER_ROW + 's per row | Sampling: ' + SAMPLING_RATE + 'Hz', 450, legendY + 4);

        var partSuffix = numParts > 1 ? '_part' + (partIdx + 1) : '';
        var dataURL = exportCanvas.toDataURL('image/' + format, 0.95);
        var link = document.createElement('a');
        link.download = 'ecg_complete_' + timestamp.replace(/[:.]/g, '-') + partSuffix + '.' + format;
        link.href = dataURL;
        link.click();

        console.log('[ECG] Exported part ' + (partIdx + 1) + '/' + numParts + ' as ' + format.toUpperCase());
    }

    console.log('[ECG] Export complete: ' + numParts + ' file(s) generated');
}

function formatTime(seconds) {
    var mins = Math.floor(seconds / 60);
    var secs = (seconds % 60).toFixed(1);
    if (mins > 0) {
        return mins + ':' + secs.padStart(4, '0');
    }
    return secs + 's';
}

// ============================================================
// AUTO-BATCH FUNCTIONS
// ============================================================

function checkAutoBatch() {
    if (!autoBatchEnabled) return;

    var unsavedSamples = currentIndex - lastBatchEndSample;
    if (unsavedSamples >= AUTO_BATCH_INTERVAL_SAMPLES) {
        saveBatch(lastBatchEndSample, currentIndex);
    }
}

function saveBatch(startSample, endSample) {
    if (endSample <= startSample) return;

    var batchNum = savedBatches.length + 1;
    console.log('[ECG] Auto-saving batch ' + batchNum + ': samples ' + startSample + ' to ' + endSample);

    var batchCanvas = generateBatchCanvas(startSample, endSample, batchNum);
    var dataURL = batchCanvas.toDataURL('image/png', 0.95);

    savedBatches.push({
        batchNum: batchNum,
        startSample: startSample,
        endSample: endSample,
        startTime: startSample / SAMPLING_RATE,
        endTime: endSample / SAMPLING_RATE,
        dataURL: dataURL,
        timestamp: new Date().toISOString()
    });

    lastBatchEndSample = endSample;
    updateBatchStatus();

    console.log('[ECG] Batch ' + batchNum + ' saved (' + ((endSample - startSample) / SAMPLING_RATE).toFixed(1) + 's)');
}

function generateBatchCanvas(startSample, endSample, batchNum) {
    var EXPORT_MAX_WIDTH = 10000;
    var ROW_HEIGHT = 250;
    var HEADER_HEIGHT = 90;
    var FOOTER_HEIGHT = 80;
    var SECONDS_PER_ROW = 30;

    var totalSamples = endSample - startSample;
    var samplesPerRow = Math.round(SECONDS_PER_ROW * SAMPLING_RATE);
    var numRows = Math.ceil(totalSamples / samplesPerRow);

    var exportWidth = EXPORT_MAX_WIDTH;
    var exportHeight = HEADER_HEIGHT + numRows * ROW_HEIGHT + FOOTER_HEIGHT;

    var exportCanvas = document.createElement('canvas');
    exportCanvas.width = exportWidth;
    exportCanvas.height = exportHeight;
    var exportCtx = exportCanvas.getContext('2d');

    exportCtx.fillStyle = '#ffffff';
    exportCtx.fillRect(0, 0, exportWidth, exportHeight);

    var buffer = [];
    for (var i = startSample; i < endSample && i < ecgData.length; i++) {
        buffer.push(ecgData[i]);
    }

    var batchGlobalMinVal = Math.min.apply(null, buffer);
    var batchGlobalMaxVal = Math.max.apply(null, buffer);
    var batchGlobalRange = batchGlobalMaxVal - batchGlobalMinVal || 1;

    var modelName = document.getElementById('modelName').textContent;
    var timestamp = new Date().toISOString();

    exportCtx.fillStyle = '#333333';
    exportCtx.font = 'bold 18px Arial';
    exportCtx.fillText('ECG Recording - Batch ' + batchNum, 20, 30);

    exportCtx.font = '12px Arial';
    exportCtx.fillStyle = '#666666';
    exportCtx.fillText('Model: ' + modelName, 20, 50);
    exportCtx.fillText('Saved: ' + timestamp, 20, 68);

    var batchStartTime = (startSample / SAMPLING_RATE).toFixed(2);
    var batchEndTime = (endSample / SAMPLING_RATE).toFixed(2);
    exportCtx.fillText('Time Range: ' + batchStartTime + 's - ' + batchEndTime + 's | ' + numRows + ' row(s)', 300, 50);

    for (var rowIdx = 0; rowIdx < numRows; rowIdx++) {
        var rowStartSample = rowIdx * samplesPerRow;
        var rowEndSample = Math.min(rowStartSample + samplesPerRow, totalSamples);

        var rowBuffer = buffer.slice(rowStartSample, rowEndSample);
        if (rowBuffer.length === 0) continue;

        var graphX = 100;
        var graphY = HEADER_HEIGHT + rowIdx * ROW_HEIGHT + 30;
        var graphWidth = exportWidth - 120;
        var graphHeight = ROW_HEIGHT - 50;

        var rowStartTime = ((startSample + rowStartSample) / SAMPLING_RATE);
        var rowEndTime = ((startSample + rowEndSample) / SAMPLING_RATE);

        exportCtx.fillStyle = '#1a5276';
        exportCtx.font = 'bold 14px Arial';
        exportCtx.fillText(formatTime(rowStartTime), 10, graphY + graphHeight / 2 + 5);
        exportCtx.fillText(formatTime(rowEndTime), exportWidth - 85, graphY + graphHeight / 2 + 5);

        exportCtx.fillStyle = '#7f8c8d';
        exportCtx.font = '10px Arial';
        exportCtx.fillText('Row ' + (rowIdx + 1), 10, graphY - 5);

        exportCtx.strokeStyle = '#cccccc';
        exportCtx.lineWidth = 1;
        exportCtx.strokeRect(graphX, graphY, graphWidth, graphHeight);

        // Medical grid
        exportCtx.strokeStyle = '#ffcccc';
        exportCtx.lineWidth = 0.5;
        for (var gx = graphX; gx <= graphX + graphWidth; gx += 15) {
            exportCtx.beginPath();
            exportCtx.moveTo(gx, graphY);
            exportCtx.lineTo(gx, graphY + graphHeight);
            exportCtx.stroke();
        }
        for (var gy = graphY; gy <= graphY + graphHeight; gy += 15) {
            exportCtx.beginPath();
            exportCtx.moveTo(graphX, gy);
            exportCtx.lineTo(graphX + graphWidth, gy);
            exportCtx.stroke();
        }

        // Draw ECG signal
        if (rowBuffer.length >= 2) {
            exportCtx.strokeStyle = '#00aa66';
            exportCtx.lineWidth = 1.5;
            exportCtx.beginPath();

            for (var ri = 0; ri < rowBuffer.length; ri++) {
                var rx = graphX + (ri / samplesPerRow) * graphWidth;
                var ry = graphY + graphHeight - ((rowBuffer[ri] - batchGlobalMinVal) / batchGlobalRange) * (graphHeight - 20) - 10;

                if (ri === 0) {
                    exportCtx.moveTo(rx, ry);
                } else {
                    exportCtx.lineTo(rx, ry);
                }
            }
            exportCtx.stroke();

            // Draw R-peak markers
            annotations.forEach(function(ann) {
                var globalIdx = ann.sample_index - startSample;
                if (globalIdx >= rowStartSample && globalIdx < rowEndSample) {
                    var localIdx = globalIdx - rowStartSample;
                    if (localIdx >= 0 && localIdx < rowBuffer.length) {
                        var ax = graphX + (localIdx / samplesPerRow) * graphWidth;
                        var ay = graphY + graphHeight - ((rowBuffer[localIdx] - batchGlobalMinVal) / batchGlobalRange) * (graphHeight - 20) - 10;

                        var classResult = classifications.find(function(c) { return c.r_peak === ann.sample_index; });
                        if (classResult && classResult.correct === false) {
                            exportCtx.strokeStyle = '#cc8800';
                            exportCtx.lineWidth = 2;
                            exportCtx.beginPath();
                            exportCtx.arc(ax, ay, 6, 0, Math.PI * 2);
                            exportCtx.stroke();
                        }

                        exportCtx.fillStyle = ann.beat_type === 'N' ? '#00aa66' : '#cc3333';
                        exportCtx.beginPath();
                        exportCtx.arc(ax, ay, 3, 0, Math.PI * 2);
                        exportCtx.fill();
                    }
                }
            });
        }
    }

    // Legend
    var legendY = exportHeight - 50;
    exportCtx.font = '11px Arial';
    exportCtx.fillStyle = '#00aa66';
    exportCtx.beginPath();
    exportCtx.arc(60, legendY, 5, 0, Math.PI * 2);
    exportCtx.fill();
    exportCtx.fillStyle = '#333333';
    exportCtx.fillText('Normal', 72, legendY + 4);

    exportCtx.fillStyle = '#cc3333';
    exportCtx.beginPath();
    exportCtx.arc(140, legendY, 5, 0, Math.PI * 2);
    exportCtx.fill();
    exportCtx.fillStyle = '#333333';
    exportCtx.fillText('Abnormal', 152, legendY + 4);

    exportCtx.strokeStyle = '#cc8800';
    exportCtx.lineWidth = 2;
    exportCtx.beginPath();
    exportCtx.arc(240, legendY, 7, 0, Math.PI * 2);
    exportCtx.stroke();
    exportCtx.fillStyle = '#333333';
    exportCtx.fillText('False', 255, legendY + 4);

    exportCtx.fillStyle = '#666666';
    exportCtx.font = '10px Arial';
    exportCtx.fillText('Sampling: ' + SAMPLING_RATE + 'Hz | ' + SECONDS_PER_ROW + 's/row', 320, legendY + 4);

    return exportCanvas;
}

function updateBatchStatus() {
    var statusEl = document.getElementById('batchStatus');
    if (!statusEl) return;

    var totalSaved = savedBatches.reduce(function(sum, b) { return sum + (b.endSample - b.startSample); }, 0);
    var savedSeconds = totalSaved / SAMPLING_RATE;
    var unsavedSeconds = (currentIndex - lastBatchEndSample) / SAMPLING_RATE;

    var html = '<span style="color: #00ff88;">' + savedBatches.length + ' batch' + (savedBatches.length !== 1 ? 'es' : '') + '</span>';
    html += '<span style="color: #888; margin-left: 10px;">(' + savedSeconds.toFixed(0) + 's saved)</span>';
    if (unsavedSeconds > 10) {
        html += '<span style="color: #ffd700; margin-left: 10px;">' + unsavedSeconds.toFixed(0) + 's pending</span>';
    }
    statusEl.innerHTML = html;
}

async function downloadAllBatches() {
    if (savedBatches.length === 0) {
        alert('No batches saved yet. Recording auto-saves batches every 2 minutes.');
        return;
    }

    var totalBatches = savedBatches.length;
    var totalSeconds = savedBatches.reduce(function(sum, b) { return sum + (b.endSample - b.startSample); }, 0) / SAMPLING_RATE;

    var statusEl = document.getElementById('batchStatus');
    if (statusEl) {
        statusEl.innerHTML = '<span style="color: #ffaa00;">Creating ZIP (' + totalBatches + ' batch' + (totalBatches !== 1 ? 'es' : '') + ')...</span>' +
            '<span style="color: #888; margin-left: 10px;">(' + totalSeconds.toFixed(0) + 's total)</span>';
    }

    console.log('[ECG] Creating ZIP with ' + totalBatches + ' batches...');

    try {
        var zip = new JSZip();

        for (var i = 0; i < savedBatches.length; i++) {
            var batch = savedBatches[i];
            var base64Data = batch.dataURL.split(',')[1];
            var filename = 'ecg_batch_' + batch.batchNum + '_' + batch.timestamp.replace(/[:.]/g, '-') + '.png';
            zip.file(filename, base64Data, {base64: true});
        }

        var zipBlob = await zip.generateAsync({type: 'blob'});

        var dlTimestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
        var link = document.createElement('a');
        link.download = 'ecg_recording_' + dlTimestamp + '.zip';
        link.href = URL.createObjectURL(zipBlob);
        link.click();

        setTimeout(function() { URL.revokeObjectURL(link.href); }, 1000);

        console.log('[ECG] ZIP downloaded successfully');

        if (statusEl) {
            statusEl.innerHTML = '<span style="color: #00ff88;">ZIP downloaded (' + totalBatches + ' batch' + (totalBatches !== 1 ? 'es' : '') + ')!</span>' +
                '<span style="color: #888; margin-left: 10px;">(' + totalSeconds.toFixed(0) + 's total)</span>';
        }
    } catch (error) {
        console.error('[ECG] Error creating ZIP:', error);
        alert('Error creating ZIP file. Please try again.');
        if (statusEl) {
            statusEl.innerHTML = '<span style="color: #ff4444;">Error creating ZIP</span>';
        }
    }
}

function exportUnsaved(format) {
    format = format || 'png';
    var unsavedStart = lastBatchEndSample;
    var unsavedEnd = currentIndex;

    if (unsavedEnd <= unsavedStart) {
        alert('No unsaved data to export. All data has been saved in batches.');
        return;
    }

    console.log('[ECG] Exporting unsaved data: ' + unsavedStart + ' to ' + unsavedEnd);

    var batchCanvas = generateBatchCanvas(unsavedStart, unsavedEnd, savedBatches.length + 1);
    var dataURL = batchCanvas.toDataURL('image/' + format, 0.95);
    var link = document.createElement('a');
    var timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    link.download = 'ecg_unsaved_' + timestamp + '.' + format;
    link.href = dataURL;
    link.click();

    console.log('[ECG] Unsaved data exported');
}

function forceSaveBatch() {
    var unsavedSamples = currentIndex - lastBatchEndSample;
    if (unsavedSamples < MIN_BATCH_SAMPLES) {
        alert('Need at least ' + MIN_BATCH_SECONDS + ' seconds of unsaved data to create a batch.');
        return;
    }
    saveBatch(lastBatchEndSample, currentIndex);
}

// Draw beat waveform on the beat snapshot canvas
function drawBeatWaveform(waveform, isAbnormal) {
    isAbnormal = isAbnormal || false;
    var width = beatCanvas.getBoundingClientRect().width;
    var height = beatCanvas.getBoundingClientRect().height;

    beatCtx.fillStyle = '#0a0a1a';
    beatCtx.fillRect(0, 0, width, height);

    // Draw grid
    beatCtx.strokeStyle = 'rgba(0, 255, 136, 0.1)';
    beatCtx.lineWidth = 1;
    for (var gx = 0; gx < width; gx += BATCH_GRID_SPACING) {
        beatCtx.beginPath();
        beatCtx.moveTo(gx, 0);
        beatCtx.lineTo(gx, height);
        beatCtx.stroke();
    }
    for (var gy = 0; gy < height; gy += BATCH_GRID_SPACING) {
        beatCtx.beginPath();
        beatCtx.moveTo(0, gy);
        beatCtx.lineTo(width, gy);
        beatCtx.stroke();
    }

    if (!waveform || waveform.length < 2) return;

    var minVal = Math.min.apply(null, waveform);
    var maxVal = Math.max.apply(null, waveform);
    var range = maxVal - minVal || 1;

    beatCtx.strokeStyle = isAbnormal ? '#ff4757' : '#00ff88';
    beatCtx.lineWidth = 2;
    beatCtx.beginPath();

    for (var i = 0; i < waveform.length; i++) {
        var x = (i / waveform.length) * width;
        var y = height - ((waveform[i] - minVal) / range) * (height - 20) - 10;

        if (i === 0) {
            beatCtx.moveTo(x, y);
        } else {
            beatCtx.lineTo(x, y);
        }
    }
    beatCtx.stroke();

    // Draw R-peak marker
    var rPeakX = (currentRPeakPos / waveform.length) * width;
    var rPeakY = height - ((waveform[Math.min(currentRPeakPos, waveform.length - 1)] - minVal) / range) * (height - 20) - 10;
    beatCtx.fillStyle = '#ffcc00';
    beatCtx.beginPath();
    beatCtx.arc(rPeakX, rPeakY, 6, 0, Math.PI * 2);
    beatCtx.fill();
    beatCtx.fillStyle = '#ffcc00';
    beatCtx.font = '11px Arial';
    beatCtx.fillText('R-peak', rPeakX - 18, rPeakY - 10);
}

// Load data from server
async function loadData() {
    var response = await fetch('/api/data');
    var data = await response.json();
    ecgData = data.signal;
    annotations = data.annotations;
    console.log('Loaded ' + ecgData.length + ' ECG samples and ' + annotations.length + ' annotations');

    globalMinVal = Infinity;
    globalMaxVal = -Infinity;
}

// Draw ECG signal
function drawECG() {
    var width = canvas.getBoundingClientRect().width;
    var height = canvas.getBoundingClientRect().height;

    ctx.fillStyle = '#0a0a1a';
    ctx.fillRect(0, 0, width, height);

    // Draw grid
    ctx.strokeStyle = 'rgba(0, 255, 136, 0.1)';
    ctx.lineWidth = 1;
    for (var gx = 0; gx < width; gx += 50) {
        ctx.beginPath();
        ctx.moveTo(gx, 0);
        ctx.lineTo(gx, height);
        ctx.stroke();
    }
    for (var gy = 0; gy < height; gy += 50) {
        ctx.beginPath();
        ctx.moveTo(0, gy);
        ctx.lineTo(width, gy);
        ctx.stroke();
    }

    // Calculate display range based on view offset
    var endSample = isLive ? currentIndex : Math.max(0, currentIndex + Math.round(viewOffset * SAMPLING_RATE));
    var startSample = Math.max(0, endSample - DISPLAY_SAMPLES);

    var buffer = [];
    for (var i = startSample; i < endSample && i < ecgData.length; i++) {
        buffer.push(ecgData[i]);
    }

    if (buffer.length < 2) return;

    var localMinVal = Math.min.apply(null, buffer);
    var localMaxVal = Math.max.apply(null, buffer);

    if (localMinVal < globalMinVal) globalMinVal = localMinVal;
    if (localMaxVal > globalMaxVal) globalMaxVal = localMaxVal;

    var minVal = globalMinVal;
    var maxVal = globalMaxVal;
    var range = maxVal - minVal || 1;

    // Dynamic height expansion
    var visibleAnnotations = 0;
    annotations.forEach(function(ann) {
        if (ann.sample_index > startSample && ann.sample_index <= endSample) {
            visibleAnnotations++;
        }
    });

    var baseHeight = MIN_GRAPH_HEIGHT;
    var heightPerAnnotation = 5;
    var annotationBonus = Math.min(visibleAnnotations * heightPerAnnotation, 200);
    var desiredHeight = baseHeight + annotationBonus;

    updateGraphHeight(desiredHeight);

    // Draw ECG line
    ctx.strokeStyle = '#00ff88';
    ctx.lineWidth = 2;
    ctx.beginPath();

    for (var di = 0; di < buffer.length; di++) {
        var dx = (di / DISPLAY_SAMPLES) * width;
        var dy = height - ((buffer[di] - minVal) / range) * (height - 40) - 20;

        if (di === 0) {
            ctx.moveTo(dx, dy);
        } else {
            ctx.lineTo(dx, dy);
        }
    }
    ctx.stroke();

    // Draw R-peak markers
    annotations.forEach(function(ann) {
        if (ann.sample_index > startSample && ann.sample_index <= endSample) {
            var bufferIdx = ann.sample_index - startSample;
            if (bufferIdx >= 0 && bufferIdx < buffer.length) {
                var ax = (bufferIdx / DISPLAY_SAMPLES) * width;
                var ay = height - ((buffer[bufferIdx] - minVal) / range) * (height - 40) - 20;

                var classResult = classifications.find(function(c) { return c.r_peak === ann.sample_index; });
                if (classResult && classResult.correct === false) {
                    ctx.strokeStyle = '#ffd700';
                    ctx.lineWidth = 3;
                    ctx.beginPath();
                    ctx.arc(ax, ay, 10, 0, Math.PI * 2);
                    ctx.stroke();
                }

                ctx.fillStyle = ann.beat_type === 'N' ? '#00ff88' : '#ff4757';
                ctx.beginPath();
                ctx.arc(ax, ay, 6, 0, Math.PI * 2);
                ctx.fill();
            }
        }
    });

    // Show "History Mode" indicator if not live
    if (!isLive) {
        ctx.fillStyle = 'rgba(255, 215, 0, 0.9)';
        ctx.font = 'bold 14px Arial';
        ctx.fillText('VIEWING HISTORY', 10, 25);
    }
}

// Update time display
function updateTime() {
    var displayIndex = isLive ? currentIndex : Math.max(0, currentIndex + Math.round(viewOffset * SAMPLING_RATE));
    var seconds = displayIndex / SAMPLING_RATE;
    var minutes = Math.floor(seconds / 60);
    var secs = (seconds % 60).toFixed(3);
    document.getElementById('currentTime').textContent =
        minutes + ':' + secs.padStart(6, '0');
}

// Calculate BPM from recent beat intervals
function calculateBPM(currentBeatSample) {
    beatTimes.push(currentBeatSample);

    if (beatTimes.length > 10) {
        beatTimes.shift();
    }

    if (beatTimes.length < 2) return null;

    var totalInterval = 0;
    var count = 0;
    for (var i = 1; i < beatTimes.length; i++) {
        var interval = (beatTimes[i] - beatTimes[i - 1]) / SAMPLING_RATE;
        if (interval > 0.3 && interval < 2.0) {
            totalInterval += interval;
            count++;
        }
    }

    if (count === 0) return null;

    var avgInterval = totalInterval / count;
    return Math.round(60 / avgInterval);
}

// Check for beats and classify
async function checkForBeats() {
    if (isClassifying) return;

    var samplesToCheck = Math.max(1, Math.round(speedMultiplier * (SAMPLING_RATE / 60)));
    var prevSample = currentIndex - samplesToCheck;

    var beatsToClassify = [];
    for (var i = 0; i < annotations.length; i++) {
        var ann = annotations[i];
        if (ann.sample_index > prevSample && ann.sample_index <= currentIndex &&
            ann.beat_type !== '+' && !processedBeats.has(ann.sample_index)) {
            beatsToClassify.push(ann);
        }
    }

    if (beatsToClassify.length === 0) return;

    isClassifying = true;

    try {
        for (var bi = 0; bi < beatsToClassify.length; bi++) {
            var beat = beatsToClassify[bi];
            processedBeats.add(beat.sample_index);

            if (processedBeats.size > 5000) {
                var toRemove = Array.from(processedBeats).slice(0, 1000);
                toRemove.forEach(function(v) { processedBeats.delete(v); });
            }

            try {
                var response = await fetch('/api/classify', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        r_peak: beat.sample_index,
                        beat_type: beat.beat_type
                    })
                });
                var result = await response.json();
                console.log('[ECG] Beat at', beat.sample_index, ':', result.predicted);
                addClassification(result);

                var bpm = calculateBPM(beat.sample_index);
                if (bpm !== null && bpm > 0 && bpm < 300) {
                    document.getElementById('heartRate').textContent = bpm;
                }
            } catch (e) {
                console.error('Classification error:', e);
            }
        }
    } finally {
        isClassifying = false;
    }
}

// Add classification result
function addClassification(result) {
    classifications.unshift(result);

    if (classifications.length > MAX_CLASSIFICATIONS) {
        classifications = classifications.slice(0, MAX_CLASSIFICATIONS);
    }

    if (result.correct === false) {
        falseDetections.unshift(result);
        if (falseDetections.length > MAX_FALSE_DETECTIONS) {
            falseDetections = falseDetections.slice(0, MAX_FALSE_DETECTIONS);
        }
        updateFalseDetectionList();
    }

    var total = classifications.filter(function(c) { return c.correct !== null; }).length;
    var normal = classifications.filter(function(c) { return c.predicted === 'NORMAL'; }).length;
    var abnormal = classifications.filter(function(c) { return c.predicted === 'ABNORMAL'; }).length;
    var correct = classifications.filter(function(c) { return c.correct === true; }).length;
    var known = classifications.filter(function(c) { return c.correct !== null; }).length;

    document.getElementById('totalBeats').textContent = classifications.length;
    document.getElementById('normalBeats').textContent = normal;
    document.getElementById('abnormalBeats').textContent = abnormal;
    document.getElementById('falseCount').textContent = falseDetections.length;
    if (known > 0) {
        document.getElementById('accuracy').textContent =
            Math.round((correct / known) * 100) + '%';
    }

    var statusEl = document.getElementById('currentStatus');
    statusEl.textContent = result.predicted;
    statusEl.className = 'value ' + result.predicted.toLowerCase();

    var prob = result.probability;
    var probBar = document.getElementById('probBar');
    probBar.style.width = (prob * 100) + '%';
    probBar.style.background = prob >= 0.5 ? '#ff4757' : '#00ff88';
    document.getElementById('probText').textContent =
        'Abnormal Probability: ' + (prob * 100).toFixed(1) + '%';

    // Update beat snapshot display
    if (result.beat_waveform) {
        currentBeatWaveform = result.beat_waveform;
        currentRPeakPos = result.r_peak_pos_in_beat || 70;
        currentBeatLength = result.beat_length || 188;
        var isAbnormal = result.predicted === 'ABNORMAL';
        drawBeatWaveform(result.beat_waveform, isAbnormal);

        var beatTypeEl = document.getElementById('beatTypeDisplay');
        beatTypeEl.textContent = result.beat_type;
        beatTypeEl.style.color = result.beat_type === 'N' ? '#00ff88' : '#ff4757';

        var groundTruthEl = document.getElementById('groundTruthDisplay');
        groundTruthEl.textContent = result.ground_truth;
        groundTruthEl.style.color = result.ground_truth === 'NORMAL' ? '#00ff88' : '#ff4757';

        var predictionEl = document.getElementById('predictionDisplay');
        predictionEl.textContent = result.predicted;
        predictionEl.style.color = result.predicted === 'NORMAL' ? '#00ff88' : '#ff4757';
    }

    // Update classification list
    var listEl = document.getElementById('classificationList');
    if (classifications.length === 1) {
        listEl.innerHTML = '';
    }

    var time = (result.r_peak / SAMPLING_RATE).toFixed(2);
    var item = document.createElement('div');
    item.className = 'classification-item ' + result.predicted.toLowerCase();
    if (result.correct === false) item.style.border = '2px solid #ffd700';
    item.style.cursor = 'pointer';
    item.onclick = function() { navigateToTime(result.r_peak); };
    item.innerHTML =
        '<div class="beat-info">' +
            '<div>Beat Type: ' + result.beat_type + ' -> ' + result.predicted + '</div>' +
            '<div class="beat-time">Time: ' + time + 's | Prob: ' + (result.probability * 100).toFixed(1) + '%</div>' +
        '</div>' +
        '<span class="prediction-badge ' + result.predicted.toLowerCase() + '">' + result.predicted + '</span>';
    listEl.insertBefore(item, listEl.firstChild);

    while (listEl.children.length > 100) {
        listEl.removeChild(listEl.lastChild);
    }
}

// Update false detection list
function updateFalseDetectionList() {
    var listEl = document.getElementById('falseDetectionList');

    if (falseDetections.length === 0) {
        listEl.innerHTML = '<p style="color: #888; text-align: center;">No false detections yet.</p>';
        return;
    }

    listEl.innerHTML = '';

    falseDetections.slice(0, 50).forEach(function(result) {
        var time = (result.r_peak / SAMPLING_RATE).toFixed(2);
        var item = document.createElement('div');
        item.style.cssText = 'display: flex; justify-content: space-between; align-items: center; padding: 8px 10px; margin-bottom: 6px; border-radius: 8px; background: rgba(255, 215, 0, 0.15); border-left: 3px solid #ffd700; cursor: pointer;';
        item.onclick = function() { navigateToTime(result.r_peak); };
        item.innerHTML =
            '<div>' +
                '<span style="color: #ffd700; font-weight: bold;">' + time + 's</span>' +
                '<span style="color: #aaa; font-size: 11px; margin-left: 8px;">Expected: ' + result.ground_truth + ' | Got: ' + result.predicted + '</span>' +
            '</div>';
        item.onmouseover = function() { item.style.background = 'rgba(255, 215, 0, 0.3)'; item.style.transform = 'translateX(5px)'; };
        item.onmouseout = function() { item.style.background = 'rgba(255, 215, 0, 0.15)'; item.style.transform = 'none'; };
        listEl.appendChild(item);
    });
}

// Animation loop with proper timing
let lastFrameTime = 0;
const targetFPS = 60;
const frameInterval = 1000 / targetFPS;
let lastBatchCheckTime = 0;

function animate(timestamp) {
    if (!isRunning) return;

    var deltaTime = timestamp - lastFrameTime;

    if (deltaTime >= frameInterval) {
        lastFrameTime = timestamp - (deltaTime % frameInterval);

        var samplesPerSecond = SAMPLING_RATE * speedMultiplier;
        var samplesToAdvance = Math.max(1, Math.round(samplesPerSecond / targetFPS));

        for (var i = 0; i < samplesToAdvance; i++) {
            if (currentIndex < ecgData.length) {
                currentIndex++;
            }
        }

        if (isLive) {
            drawECG();
            updateTime();
        }

        checkForBeats();

        if (timestamp - lastBatchCheckTime > BATCH_CHECK_INTERVAL_MS) {
            lastBatchCheckTime = timestamp;
            checkAutoBatch();
            updateBatchStatus();
        }
    }

    if (currentIndex < ecgData.length) {
        animationId = requestAnimationFrame(animate);
    } else {
        isRunning = false;
        document.getElementById('currentStatus').textContent = 'Complete!';
        if (currentIndex - lastBatchEndSample > MIN_BATCH_SAMPLES) {
            saveBatch(lastBatchEndSample, currentIndex);
        }
    }
}

// Control functions
async function startSimulation() {
    if (ecgData.length === 0) {
        await loadData();
    }
    isRunning = true;
    lastFrameTime = performance.now();
    animationId = requestAnimationFrame(animate);
}

function stopSimulation() {
    isRunning = false;
    if (animationId) {
        cancelAnimationFrame(animationId);
    }

    var unsavedSamples = currentIndex - lastBatchEndSample;
    if (unsavedSamples >= MIN_BATCH_SAMPLES) {
        console.log('[ECG] Auto-saving final batch on stop...');
        saveBatch(lastBatchEndSample, currentIndex);
    }

    updateBatchStatus();
}

async function resetSimulation() {
    stopSimulation();
    currentIndex = 0;
    classifications = [];
    falseDetections = [];
    beatTimes = [];
    currentBeatWaveform = null;
    viewOffset = 0;
    isLive = true;

    isClassifying = false;
    classificationQueue = [];
    processedBeats.clear();

    globalMinVal = Infinity;
    globalMaxVal = -Infinity;

    maxGraphHeight = MIN_GRAPH_HEIGHT;

    try {
        var resp = await fetch('/api/reset', { method: 'POST' });
        if (!resp.ok) {
            console.warn('Backend reset returned non-OK status');
        }
    } catch (e) {
        console.error('Failed to reset backend:', e);
        alert('Warning: Backend reset failed. Please refresh the page if issues persist.');
    }

    savedBatches = [];
    lastBatchEndSample = 0;
    updateBatchStatus();

    document.getElementById('totalBeats').textContent = '0';
    document.getElementById('normalBeats').textContent = '0';
    document.getElementById('abnormalBeats').textContent = '0';
    document.getElementById('accuracy').textContent = '--';
    document.getElementById('heartRate').textContent = '--';
    document.getElementById('falseCount').textContent = '0';
    document.getElementById('currentStatus').textContent = 'Waiting...';
    document.getElementById('currentStatus').className = 'value';
    document.getElementById('probBar').style.width = '0%';
    document.getElementById('probText').textContent = 'Abnormal Probability: --';
    document.getElementById('classificationList').innerHTML =
        '<p style="color: #888; text-align: center;">No classifications yet. Start the simulation!</p>';
    document.getElementById('falseDetectionList').innerHTML =
        '<p style="color: #888; text-align: center;">No false detections yet.</p>';
    document.getElementById('currentTime').textContent = '0:00.000';

    updateHistoryUI();

    document.getElementById('beatTypeDisplay').textContent = '--';
    document.getElementById('beatTypeDisplay').style.color = '#00ff88';
    document.getElementById('groundTruthDisplay').textContent = '--';
    document.getElementById('groundTruthDisplay').style.color = '#00ff88';
    document.getElementById('predictionDisplay').textContent = '--';
    document.getElementById('predictionDisplay').style.color = '#00ff88';

    var width = beatCanvas.getBoundingClientRect().width;
    var height = beatCanvas.getBoundingClientRect().height;
    beatCtx.fillStyle = '#0a0a1a';
    beatCtx.fillRect(0, 0, width, height);

    drawECG();
}

// Load model info
async function loadModelInfo() {
    try {
        var response = await fetch('/api/model_info');
        var info = await response.json();
        document.getElementById('modelName').textContent = info.name;
    } catch (e) {
        console.error('Failed to load model info:', e);
    }
}

// Initialize
loadModelInfo();
loadData().then(function() {
    drawECG();
});

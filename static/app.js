/* =========================================================
   Meeting AI — Frontend Logic
   
   TWO separate flows:
   
   1. FILE UPLOAD  (unchanged from before)
      User picks a file -> POST /api/transcribe -> displayTranscript()
   
   2. LIVE RECORDING  (new: real WebSocket streaming)
      User clicks Start -> AudioContext (16kHz) -> ScriptProcessorNode
      -> Int16 PCM bytes -> WebSocket /ws/live -> JSON chunks arrive
      -> appendLiveChunk() adds blocks in real-time as you speak
      -> User clicks Stop -> WebSocket closes -> liveSummaryBtn shown
   ========================================================= */

'use strict';

// ── Shared state ─────────────────────────────────────────────────────────────
let currentTranscriptData = null;   // set by file upload, used by batch summary
let liveSpeakerColors = {};
let liveColorIndex = 1;
let liveChunks = [];     // accumulates chunks for summary after stop

// ── Speaker color helper ──────────────────────────────────────────────────────
function getSpeakerColorClass(speakerLabel) {
    if (!liveSpeakerColors[speakerLabel]) {
        liveSpeakerColors[speakerLabel] = `speaker-color-${liveColorIndex}`;
        liveColorIndex = liveColorIndex >= 5 ? 1 : liveColorIndex + 1;
    }
    return liveSpeakerColors[speakerLabel];
}

// ─────────────────────────────────────────────────────────────────────────────
//  FLOW 1 — File Upload
// ─────────────────────────────────────────────────────────────────────────────

document.getElementById('audioFile').addEventListener('change', function (e) {
    const fileName = e.target.files[0]?.name || 'Click to browse or drag a file here';
    document.getElementById('file-name').textContent = fileName;
    document.getElementById('transcribeBtn').disabled = !e.target.files.length;
});

document.getElementById('uploadForm').addEventListener('submit', async function (e) {
    e.preventDefault();
    const file = document.getElementById('audioFile').files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    await submitToBackend(formData);
});

async function submitToBackend(formData) {
    document.getElementById('uploadForm').classList.add('hidden');
    document.querySelector('.recording-container').classList.add('hidden');
    document.querySelector('.divider').classList.add('hidden');
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('resultsSection').classList.add('hidden');
    document.getElementById('summaryContainer').classList.add('hidden');
    document.getElementById('liveSection').classList.add('hidden');

    try {
        const response = await fetch('/api/transcribe', { method: 'POST', body: formData });
        if (!response.ok) throw new Error(`Server ${response.status}: ${await response.text()}`);

        const data = await response.json();
        currentTranscriptData = data.transcript;

        displayTranscript(data);

        document.getElementById('loading').classList.add('hidden');
        document.getElementById('uploadForm').classList.remove('hidden');
        document.querySelector('.recording-container').classList.remove('hidden');
        document.querySelector('.divider').classList.remove('hidden');
        document.getElementById('resultsSection').classList.remove('hidden');

    } catch (error) {
        alert('Transcription failed: ' + error.message);
        document.getElementById('loading').classList.add('hidden');
        document.getElementById('uploadForm').classList.remove('hidden');
        document.querySelector('.recording-container').classList.remove('hidden');
        document.querySelector('.divider').classList.remove('hidden');
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  FLOW 2 — Live Recording (WebSocket + AudioContext)
// ─────────────────────────────────────────────────────────────────────────────

let liveWs = null;
let audioContext = null;
let mediaStream = null;
let mediaStreamSource = null;
let scriptProcessor = null;
let recordingInterval = null;
let recordingSeconds = 0;

const startBtn = document.getElementById('startRecordBtn');
const stopBtn = document.getElementById('stopRecordBtn');
const recordingState = document.getElementById('recordingState');
const timerDisplay = document.getElementById('recordingTimer');

startBtn.addEventListener('click', async () => {
    try {
        // 1. Create AudioContext synchronously BEFORE any 'await'.
        //    Browsers will suspend the context if created after yielding to an async call.
        audioContext = new AudioContext({ sampleRate: 16000 });

        // 2. Get microphone access
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });

        // 3. Ensure context is running (sometimes still needed on Safari/Chrome)
        if (audioContext.state === 'suspended') {
            await audioContext.resume();
        }

        // 4. Create audio graph IMMEDIATELY to prevent GC or suspension
        mediaStreamSource = audioContext.createMediaStreamSource(mediaStream);
        scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);

        scriptProcessor.onaudioprocess = (e) => {
            if (!liveWs || liveWs.readyState !== WebSocket.OPEN) return;

            // Float32 [-1..1] -> Int16 [-32768..32767]
            const float32 = e.inputBuffer.getChannelData(0);
            const int16 = new Int16Array(float32.length);
            for (let i = 0; i < float32.length; i++) {
                int16[i] = Math.max(-32768, Math.min(32767, float32[i] * 32768));
            }
            liveWs.send(int16.buffer);
        };

        mediaStreamSource.connect(scriptProcessor);
        // Connect to destination (silent node) to keep the graph alive
        scriptProcessor.connect(audioContext.destination);

        // 5. Open WebSocket FIRST, stream audio only once connected.
        const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
        const wsUrl = `${wsProtocol}://${window.location.host}/ws/live`;
        liveWs = new WebSocket(wsUrl);
        liveWs.binaryType = 'arraybuffer';

        liveWs.onopen = () => {
            console.log('[WS] Connected to /ws/live');
        };

        liveWs.onmessage = (e) => {
            // Server pushes a chunk as soon as speech + transcription is ready
            const chunk = JSON.parse(e.data);
            liveChunks.push(chunk);
            appendLiveChunk(chunk);
        };

        liveWs.onerror = (e) => console.error('[WS] Error:', e);

        liveWs.onclose = () => console.log('[WS] Connection closed');

        // 4. Update UI
        resetLiveTranscript();
        document.getElementById('liveSection').classList.remove('hidden');
        document.getElementById('liveBadge').textContent = '🔴 LIVE';
        document.getElementById('liveSummaryBtn').classList.add('hidden');
        document.getElementById('liveSummaryContainer').classList.add('hidden');
        document.getElementById('resultsSection').classList.add('hidden');

        startBtn.classList.add('hidden');
        recordingState.classList.remove('hidden');

        // Timer
        recordingSeconds = 0;
        timerDisplay.textContent = '00:00';
        recordingInterval = setInterval(() => {
            recordingSeconds++;
            const m = Math.floor(recordingSeconds / 60).toString().padStart(2, '0');
            const s = (recordingSeconds % 60).toString().padStart(2, '0');
            timerDisplay.textContent = `${m}:${s}`;
        }, 1000);

    } catch (err) {
        console.error('Microphone / WebSocket error:', err);
        alert('Microphone access denied or WebSocket failed: ' + err.message);
        stopLiveRecording();
    }
});

stopBtn.addEventListener('click', () => stopLiveRecording());

function stopLiveRecording() {
    clearInterval(recordingInterval);

    // Disconnect audio graph
    if (scriptProcessor) { scriptProcessor.disconnect(); scriptProcessor = null; }
    if (audioContext) { audioContext.close(); audioContext = null; }
    if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }

    // Close WebSocket
    if (liveWs) { liveWs.close(); liveWs = null; }

    // Update UI
    startBtn.classList.remove('hidden');
    recordingState.classList.add('hidden');
    timerDisplay.textContent = '00:00';

    // Show "DONE" badge and summary button if we got any transcript
    document.getElementById('liveBadge').textContent = '✅ Done';
    if (liveChunks.length > 0) {
        document.getElementById('liveSummaryBtn').classList.remove('hidden');
    }
}

// ── Append one live chunk to the live transcript panel ────────────────────────
function appendLiveChunk(chunk) {
    const container = document.getElementById('liveTranscriptContainer');

    // Hide the "Waiting for speech…" empty state on first result
    const emptyState = document.getElementById('liveEmptyState');
    if (emptyState) emptyState.style.display = 'none';

    // Check if the last block on screen belongs to the same speaker
    const lastBlock = container.lastElementChild;
    const isSameSpeaker = lastBlock &&
        lastBlock.classList.contains('live-chunk') &&
        lastBlock.dataset.speaker === chunk.speaker;

    if (isSameSpeaker) {
        // Smart Merge: Append text to the existing block instead of creating a new one
        const textContainer = lastBlock.querySelector('.transcript-text');
        textContainer.innerHTML += ' ' + escapeHtml(chunk.text);
    } else {
        // New Speaker: Create a brand new block
        const colorClass = getSpeakerColorClass(chunk.speaker);
        const block = document.createElement('div');
        block.className = 'transcript-block live-chunk';
        block.dataset.speaker = chunk.speaker; // store for the next check
        block.innerHTML = `
            <div class="timestamp">${chunk.timestamp}</div>
            <div class="block-content">
                <div class="speaker-label ${colorClass}">${chunk.speaker}</div>
                <div class="transcript-text">${escapeHtml(chunk.text)}</div>
            </div>
        `;
        container.appendChild(block);
    }

    // Auto-scroll to latest
    container.scrollTop = container.scrollHeight;
}

function resetLiveTranscript() {
    liveChunks = [];
    liveSpeakerColors = {};
    liveColorIndex = 1;
    const container = document.getElementById('liveTranscriptContainer');
    container.innerHTML = `
        <div id="liveEmptyState" class="live-empty-state">
            <div class="spinner small"></div>
            <span>Waiting for speech…</span>
        </div>
    `;
}

function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ── Live Summary Button ───────────────────────────────────────────────────────
document.getElementById('liveSummaryBtn').addEventListener('click', async function () {
    if (!liveChunks.length) return;

    this.classList.add('hidden');
    document.getElementById('liveSummaryLoading').classList.remove('hidden');

    // Convert liveChunks to the same MeetingTranscript dict shape the batch
    // summarize endpoint expects, so we can reuse the same /api/summarize route.
    const blocks = liveChunks.map(c => ({
        speaker_label: c.speaker,
        timestamp: c.timestamp,
        start_seconds: c.start_seconds,
        end_seconds: c.start_seconds,
        text: c.text,
    }));
    const uniqueSpeakers = new Set(liveChunks.map(c => c.speaker)).size;
    const totalDuration = liveChunks.length
        ? liveChunks[liveChunks.length - 1].start_seconds
        : 0;

    try {
        const response = await fetch('/api/summarize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                transcript_dict: {
                    speaker_count: uniqueSpeakers,
                    duration_seconds: totalDuration,
                    blocks,
                },
            }),
        });

        if (!response.ok) throw new Error(`Server ${response.status}: ${await response.text()}`);

        const data = await response.json();
        const html = `
            <p><strong>Executive Summary:</strong><br>${data.executive_summary}</p><br>
            <p><strong>Key Topics:</strong><br>${data.key_topics.join('<br>')}</p><br>
            <p><strong>Decisions:</strong><br>${data.decisions.join('<br>')}</p><br>
            <p><strong>Action Items:</strong><br>${data.action_items.join('<br>')}</p>
        `;
        document.getElementById('liveSummaryContent').innerHTML = html;
        document.getElementById('liveSummaryLoading').classList.add('hidden');
        document.getElementById('liveSummaryContainer').classList.remove('hidden');

    } catch (err) {
        alert('Summarization failed: ' + err.message);
        document.getElementById('liveSummaryLoading').classList.add('hidden');
        this.classList.remove('hidden');
    }
});

// ─────────────────────────────────────────────────────────────────────────────
//  SHARED — Display transcript (file upload results)
// ─────────────────────────────────────────────────────────────────────────────

function displayTranscript(data) {
    const container = document.getElementById('transcriptContainer');
    container.innerHTML = '';

    const mins = Math.floor(data.duration_seconds / 60);
    const secs = Math.floor(data.duration_seconds % 60);
    document.getElementById('metaInfo').textContent =
        `${data.language.toUpperCase()} • ${data.transcript.speaker_count} Speakers • ${mins}:${secs.toString().padStart(2, '0')}`;

    const speakerColors = {};
    let colorIndex = 1;

    data.transcript.blocks.forEach(block => {
        if (!speakerColors[block.speaker_label]) {
            speakerColors[block.speaker_label] = `speaker-color-${colorIndex}`;
            colorIndex = colorIndex >= 5 ? 1 : colorIndex + 1;
        }

        const blockEl = document.createElement('div');
        blockEl.className = 'transcript-block';
        blockEl.innerHTML = `
            <div class="timestamp">${block.timestamp}</div>
            <div class="block-content">
                <div class="speaker-label ${speakerColors[block.speaker_label]}">${block.speaker_label}</div>
                <div class="transcript-text">${block.text}</div>
            </div>
        `;
        container.appendChild(blockEl);
    });
}

// ── Batch Summary Button ──────────────────────────────────────────────────────
document.getElementById('summaryBtn').addEventListener('click', async function () {
    if (!currentTranscriptData) return;

    this.classList.add('hidden');
    document.getElementById('summaryLoading').classList.remove('hidden');

    try {
        const response = await fetch('/api/summarize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ transcript_dict: currentTranscriptData }),
        });

        if (!response.ok) throw new Error(`Server ${response.status}: ${await response.text()}`);

        const data = await response.json();
        const summaryHtml = `
            <p><strong>Executive Summary:</strong><br>${data.executive_summary}</p><br>
            <p><strong>Key Topics:</strong><br>${data.key_topics.join('<br>')}</p><br>
            <p><strong>Decisions:</strong><br>${data.decisions.join('<br>')}</p><br>
            <p><strong>Action Items:</strong><br>${data.action_items.join('<br>')}</p>
        `;

        document.getElementById('summaryContent').innerHTML = summaryHtml;
        document.getElementById('summaryLoading').classList.add('hidden');
        document.getElementById('summaryContainer').classList.remove('hidden');

    } catch (error) {
        alert('Summarization failed: ' + error.message);
        document.getElementById('summaryLoading').classList.add('hidden');
        this.classList.remove('hidden');
    }
});

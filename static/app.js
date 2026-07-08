let currentTranscriptData = null;

document.getElementById('audioFile').addEventListener('change', function(e) {
    const fileName = e.target.files[0]?.name || 'Click to browse or drag a file here';
    document.getElementById('file-name').textContent = fileName;
    document.getElementById('transcribeBtn').disabled = !e.target.files.length;
});

document.getElementById('uploadForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const file = document.getElementById('audioFile').files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    
    await submitToBackend(formData);
});

async function submitToBackend(formData) {
    // Update UI state
    document.getElementById('uploadForm').classList.add('hidden');
    document.querySelector('.recording-container').classList.add('hidden');
    document.querySelector('.divider').classList.add('hidden');
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('resultsSection').classList.add('hidden');
    document.getElementById('summaryContainer').classList.add('hidden');

    try {
        const response = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server responded with ${response.status}: ${await response.text()}`);
        }

        const data = await response.json();
        currentTranscriptData = data.transcript;
        
        displayTranscript(data);
        
        document.getElementById('loading').classList.add('hidden');
        document.getElementById('uploadForm').classList.remove('hidden');
        document.querySelector('.recording-container').classList.remove('hidden');
        document.querySelector('.divider').classList.remove('hidden');
        document.getElementById('resultsSection').classList.remove('hidden');

    } catch (error) {
        alert("Transcription failed: " + error.message);
        document.getElementById('loading').classList.add('hidden');
        document.getElementById('uploadForm').classList.remove('hidden');
        document.querySelector('.recording-container').classList.remove('hidden');
        document.querySelector('.divider').classList.remove('hidden');
    }
}

// --- Live Recording Logic ---
let mediaRecorder;
let audioChunks = [];
let recordingInterval;
let recordingSeconds = 0;

const startBtn = document.getElementById('startRecordBtn');
const stopBtn = document.getElementById('stopRecordBtn');
const recordingState = document.getElementById('recordingState');
const timerDisplay = document.getElementById('recordingTimer');

startBtn.addEventListener('click', async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        // Supported mime types vary by browser. webm is standard for Chrome/Firefox, mp4 for Safari
        const mimeType = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 'audio/mp4';
        mediaRecorder = new MediaRecorder(stream, { mimeType });
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
            clearInterval(recordingInterval);
            
            // Stop all tracks to release the microphone hardware
            stream.getTracks().forEach(track => track.stop());

            // If stopped immediately, chunks might be empty or too small
            if (audioChunks.length === 0 || recordingSeconds < 2) {
                alert("Recording was too short. Please record for at least 2 seconds.");
                resetRecordingUI();
                return;
            }

            const audioBlob = new Blob(audioChunks, { type: mimeType });
            
            // Convert Blob to a File object so FastAPI's UploadFile accepts it seamlessly
            const fileExtension = mimeType.includes('webm') ? 'webm' : 'mp4';
            const audioFile = new File([audioBlob], `live_recording.${fileExtension}`, { type: mimeType });
            
            const formData = new FormData();
            formData.append('file', audioFile);
            
            resetRecordingUI();
            await submitToBackend(formData);
        };

        mediaRecorder.start(250); // fire ondataavailable every 250ms for safety
        
        // Update UI
        startBtn.classList.add('hidden');
        recordingState.classList.remove('hidden');
        document.getElementById('uploadForm').classList.add('hidden'); // Hide upload while recording
        
        // Start Timer
        recordingSeconds = 0;
        timerDisplay.textContent = "00:00";
        recordingInterval = setInterval(() => {
            recordingSeconds++;
            const mins = Math.floor(recordingSeconds / 60).toString().padStart(2, '0');
            const secs = (recordingSeconds % 60).toString().padStart(2, '0');
            timerDisplay.textContent = `${mins}:${secs}`;
        }, 1000);

    } catch (err) {
        console.error("Microphone error:", err);
        alert("Microphone access denied or unavailable. Please check your browser permissions.");
    }
});

stopBtn.addEventListener('click', () => {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
});

function resetRecordingUI() {
    startBtn.classList.remove('hidden');
    recordingState.classList.add('hidden');
    document.getElementById('uploadForm').classList.remove('hidden');
    timerDisplay.textContent = "00:00";
}


function displayTranscript(data) {
    const container = document.getElementById('transcriptContainer');
    container.innerHTML = '';
    
    // Formatting metadata
    const mins = Math.floor(data.duration_seconds / 60);
    const secs = Math.floor(data.duration_seconds % 60);
    document.getElementById('metaInfo').textContent = `${data.language.toUpperCase()} • ${data.transcript.speaker_count} Speakers • ${mins}:${secs.toString().padStart(2, '0')}`;

    // Colors mapper for speakers
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

document.getElementById('summaryBtn').addEventListener('click', async function() {
    if (!currentTranscriptData) return;

    this.classList.add('hidden');
    document.getElementById('summaryLoading').classList.remove('hidden');

    try {
        const response = await fetch('/api/summarize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ transcript_dict: currentTranscriptData })
        });

        if (!response.ok) {
            throw new Error(`Server responded with ${response.status}: ${await response.text()}`);
        }

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
        alert("Summarization failed: " + error.message);
        document.getElementById('summaryLoading').classList.add('hidden');
        this.classList.remove('hidden');
    }
});

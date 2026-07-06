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

    // Update UI state
    document.getElementById('uploadForm').classList.add('hidden');
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
        document.getElementById('resultsSection').classList.remove('hidden');

    } catch (error) {
        alert("Transcription failed: " + error.message);
        document.getElementById('loading').classList.add('hidden');
        document.getElementById('uploadForm').classList.remove('hidden');
    }
});

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

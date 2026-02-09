/**
 * PrecisionVoice - Frontend Application Logic
 * Handles file upload, transcription requests, and result display.
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const elements = {
        // Upload
        dropZone: document.getElementById('drop-zone'),
        fileInput: document.getElementById('file-input'),
        fileInfo: document.getElementById('file-info'),
        fileName: document.getElementById('file-name'),
        fileSize: document.getElementById('file-size'),
        clearBtn: document.getElementById('clear-btn'),
        transcribeBtn: document.getElementById('transcribe-btn'),

        // Sections
        uploadSection: document.getElementById('upload-section'),
        processingSection: document.getElementById('processing-section'),
        resultsSection: document.getElementById('results-section'),
        errorSection: document.getElementById('error-section'),

        // Processing
        processingStatus: document.getElementById('processing-status'),
        progressFill: document.getElementById('progress-fill'),
        processingTimer: document.getElementById('processing-timer'),

        // Results
        speakerCount: document.getElementById('speaker-count'),
        durationInfo: document.getElementById('duration-info'),
        processingTime: document.getElementById('processing-time'),
        transcriptContainer: document.getElementById('transcript-container'),
        downloadTxt: document.getElementById('download-txt'),
        downloadCsv: document.getElementById("download-csv"),
        newUploadBtn: document.getElementById('new-upload-btn'),

        // Error
        errorMessage: document.getElementById('error-message'),
        retryBtn: document.getElementById('retry-btn')
    };

    let selectedFile = null;

    // =====================
    // Event Listeners
    // =====================

    // Click to upload
    elements.dropZone.addEventListener('click', () => {
        elements.fileInput.click();
    });

    // File input change
    elements.fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });

    // Drag and drop
    elements.dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.dropZone.classList.add('dragover');
    });

    elements.dropZone.addEventListener('dragleave', () => {
        elements.dropZone.classList.remove('dragover');
    });

    elements.dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.dropZone.classList.remove('dragover');

        if (e.dataTransfer.files.length > 0) {
            handleFileSelection(e.dataTransfer.files[0]);
        }
    });

    // Clear file
    elements.clearBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        clearFileSelection();
    });

    // Transcribe button
    elements.transcribeBtn.addEventListener('click', () => {
        if (selectedFile) {
            startTranscription();
        }
    });

    // New upload button
    elements.newUploadBtn.addEventListener('click', resetToUpload);

    // Retry button
    elements.retryBtn.addEventListener('click', resetToUpload);

    // =====================
    // File Handling
    // =====================

    function handleFileSelection(file) {
        const allowedTypes = ['audio/mpeg', 'audio/wav', 'audio/x-wav', 'audio/mp4', 'audio/x-m4a',
            'audio/ogg', 'audio/flac', 'audio/webm', 'video/webm'];
        const allowedExtensions = ['mp3', 'wav', 'm4a', 'ogg', 'flac', 'webm'];

        // Check file extension
        const ext = file.name.split('.').pop().toLowerCase();
        if (!allowedExtensions.includes(ext)) {
            showError(`Unsupported file type: .${ext}. Supported: ${allowedExtensions.join(', ')}`);
            return;
        }

        // Check file size (100MB limit)
        const maxSize = 100 * 1024 * 1024;
        if (file.size > maxSize) {
            showError(`File too large. Maximum size: 100MB`);
            return;
        }

        selectedFile = file;

        // Update UI
        elements.fileName.textContent = file.name;
        elements.fileSize.textContent = formatFileSize(file.size);
        elements.fileInfo.classList.remove('hidden');
        elements.transcribeBtn.disabled = false;

        // Hide drop zone text
        elements.dropZone.style.display = 'none';
    }

    function clearFileSelection() {
        selectedFile = null;
        elements.fileInput.value = '';
        elements.fileInfo.classList.add('hidden');
        elements.transcribeBtn.disabled = true;
        elements.dropZone.style.display = 'block';
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // =====================
    // Transcription
    // =====================

    async function startTranscription() {
        if (!selectedFile) return;

        // Show processing UI
        showSection('processing');
        updateProgress(100, 'Processing audio... (Check server logs for details)');

        // Reset and start timer
        let seconds = 0;
        elements.processingTimer.textContent = '00:00';
        const timerInterval = setInterval(() => {
            seconds++;
            const m = Math.floor(seconds / 60);
            const s = seconds % 60;
            elements.processingTimer.textContent = `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
        }, 1000);

        try {
            const formData = new FormData();
            formData.append('file', selectedFile);

            const response = await fetch('/api/transcribe', {
                method: 'POST',
                body: formData
            });

            clearInterval(timerInterval);

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Processing failed');
            }

            const result = await response.json();
            displayResults(result);

        } catch (error) {
            clearInterval(timerInterval);
            console.error('Processing error:', error);
            showError(error.message || 'An error occurred during processing');
        }
    }

    function updateProgress(percent, status) {
        elements.progressFill.style.width = `${percent}%`;
        if (status) {
            elements.processingStatus.textContent = status;
        }
    }

    // =====================
    // Results Display
    // =====================

    function displayResults(result) {
        // ===== Metadata =====

        // Nếu chỉ dùng ROLE thì không nên hiển thị speaker_count
        if (result.roles) {
            const roleCount = Object.keys(result.roles).length;
            elements.speakerCount.textContent = `${roleCount} role${roleCount !== 1 ? 's' : ''}`;
        } else {
            elements.speakerCount.textContent = '';
        }

        elements.durationInfo.textContent = formatDuration(result.duration || 0);
        elements.processingTime.textContent = `${result.processing_time || 0}s`;

        // ===== Download links =====
        if (result.download_txt) {
            elements.downloadTxt.href = result.download_txt;
            elements.downloadTxt.style.display = 'inline-block';
        }

        if (result.download_csv) {
            elements.downloadCsv.href = result.download_csv;
            elements.downloadCsv.style.display = 'inline-block';
        }

        // ===== Render transcript =====
        renderTranscript(result.segments || []);

        // ===== Show results =====
        showSection('results');
    }

    function renderTranscript(segments) {
        elements.transcriptContainer.innerHTML = '';

        const roleColors = {};
        let colorIndex = 0;

        segments.forEach((segment) => {
            const role = segment.role || 'UNKNOWN';

            if (!(role in roleColors)) {
                colorIndex++;
                roleColors[role] = `speaker-${Math.min(colorIndex, 5)}`;
            }

            const segmentEl = document.createElement('div');
            segmentEl.className = `segment ${roleColors[role]}`;

            const start = typeof segment.start === 'number' ? segment.start : 0;
            const end = typeof segment.end === 'number' ? segment.end : 0;
            const text = segment.text ? escapeHtml(segment.text) : '';

            segmentEl.innerHTML = `
                <div class="segment-header">
                    <span class="segment-speaker">
                        ${escapeHtml(role)}
                    </span>
                    <span class="segment-time">
                        ${formatTime(start)} - ${formatTime(end)}
                    </span>
                </div>
                <p class="segment-text">${text}</p>
            `;

            elements.transcriptContainer.appendChild(segmentEl);
        });
    }



    function formatTime(seconds) {
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = Math.floor(seconds % 60);

        if (h > 0) {
            return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
        }
        return `${m}:${s.toString().padStart(2, '0')}`;
    }

    function formatDuration(seconds) {
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${m}:${s.toString().padStart(2, '0')}`;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // =====================
    // UI State Management
    // =====================

    function showSection(section) {
        elements.uploadSection.classList.add('hidden');
        elements.processingSection.classList.add('hidden');
        elements.resultsSection.classList.add('hidden');
        elements.errorSection.classList.add('hidden');

        switch (section) {
            case 'upload':
                elements.uploadSection.classList.remove('hidden');
                break;
            case 'processing':
                elements.processingSection.classList.remove('hidden');
                break;
            case 'results':
                elements.resultsSection.classList.remove('hidden');
                break;
            case 'error':
                elements.errorSection.classList.remove('hidden');
                break;
        }
    }

    function showError(message) {
        elements.errorMessage.textContent = message;
        showSection('error');
    }

    function resetToUpload() {
        clearFileSelection();
        showSection('upload');
        updateProgress(0, 'Uploading file...');
    }
});

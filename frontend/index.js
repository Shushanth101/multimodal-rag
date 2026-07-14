(function() {
  // Elements
  const chatWindow = document.getElementById('chat-window');
  const chatInput = document.getElementById('chat-input');
  const sendBtn = document.getElementById('send-btn');
  const modelSelect = document.getElementById('model-select');
  const activeModelDisplay = document.getElementById('active-model-display');
  const uploadZone = document.getElementById('upload-zone');
  const fileInput = document.getElementById('file-input');
  const uploadStatus = document.getElementById('upload-status');
  const uploadStatusText = document.getElementById('upload-status-text');
  const filesList = document.getElementById('files-list');
  const clearAllBtn = document.getElementById('clear-all-btn');
  const newChatBtn = document.getElementById('new-chat-btn');

  // Application State
  let threadId = sessionStorage.getItem('thread_id');
  if (!threadId) {
    threadId = 'session_' + Math.random().toString(36).substring(2, 11);
    sessionStorage.setItem('thread_id', threadId);
  }

  // Set up marked options if available
  if (window.marked) {
    marked.setOptions({
      gfm: true,
      breaks: true
    });
  }

  // --- Initializers ---
  function init() {
    loadFilesList();
    updateModelDisplay();
    setupEventHandlers();
  }

  // --- Event Handlers ---
  function setupEventHandlers() {
    // Model Select
    modelSelect.addEventListener('change', updateModelDisplay);

    // Auto-grow textarea
    chatInput.addEventListener('input', function() {
      this.style.height = 'auto';
      this.style.height = (this.scrollHeight - 4) + 'px';
      sendBtn.disabled = this.value.trim() === '';
    });

    // Send on Enter (but Shift+Enter makes newline)
    chatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    sendBtn.addEventListener('click', sendMessage);

    // New Chat
    newChatBtn.addEventListener('click', () => {
      threadId = 'session_' + Math.random().toString(36).substring(2, 11);
      sessionStorage.setItem('thread_id', threadId);
      chatWindow.innerHTML = `
        <div class="message ai-message">
          <div class="avatar">✦</div>
          <div class="message-content">
            <p>Started a new research thread! Ask me anything about the ingested documents.</p>
          </div>
        </div>
      `;
    });

    // Clear All Documents
    clearAllBtn.addEventListener('click', clearAllFiles);

    // Upload zone click & drag-drop
    uploadZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        uploadFiles(e.target.files);
      }
    });

    uploadZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      uploadZone.style.borderColor = 'var(--accent-color)';
    });

    uploadZone.addEventListener('dragleave', () => {
      uploadZone.style.borderColor = 'var(--border-color)';
    });

    uploadZone.addEventListener('drop', (e) => {
      e.preventDefault();
      uploadZone.style.borderColor = 'var(--border-color)';
      if (e.dataTransfer.files.length > 0) {
        uploadFiles(e.dataTransfer.files);
      }
    });
  }

  // --- Model selection helper ---
  function getSelectedModel() {
    return modelSelect.value;
  }

  function updateModelDisplay() {
    const model = getSelectedModel();
    activeModelDisplay.textContent = `Selected Model: ${model}`;
  }

  // --- Document Ingestion (API Calls) ---
  async function uploadFiles(files) {
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
    }

    uploadStatus.classList.remove('hidden');
    uploadStatusText.textContent = `Processing ${files.length} document(s)...`;

    try {
      const res = await fetch('/api/ingest', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (res.ok) {
        uploadStatusText.textContent = 'Ingestion complete!';
        setTimeout(() => uploadStatus.classList.add('hidden'), 2000);
        loadFilesList();
      } else {
        throw new Error(data.detail || 'Ingestion failed');
      }
    } catch (err) {
      alert(`Error uploading documents: ${err.message}`);
      uploadStatus.classList.add('hidden');
    }
  }

  async function loadFilesList() {
    try {
      const res = await fetch('/api/files');
      const data = await res.json();
      renderFilesList(data);
    } catch (err) {
      console.error('Error fetching files:', err);
    }
  }

  async function clearAllFiles() {
    if (!confirm('Are you sure you want to delete all ingested documents and vector embeddings? This cannot be undone.')) {
      return;
    }

    try {
      const res = await fetch('/api/files', { method: 'DELETE' });
      if (res.ok) {
        loadFilesList();
        // Append sys message
        appendSystemMessage('All ingested documents and vector databases have been cleared.');
      } else {
        const data = await res.json();
        alert(`Error clearing documents: ${data.detail}`);
      }
    } catch (err) {
      alert(`Error clearing documents: ${err.message}`);
    }
  }

  function renderFilesList(files) {
    if (!files || files.length === 0) {
      filesList.className = 'empty-list-placeholder';
      filesList.innerHTML = 'No documents ingested yet. Upload a PDF to start asking questions.';
      return;
    }

    filesList.className = '';
    filesList.innerHTML = '';
    files.forEach(file => {
      const item = document.createElement('div');
      item.className = 'file-item';
      item.innerHTML = `
        <div class="file-info">
          <span class="file-name" title="${file.filename}">${file.filename}</span>
          <span class="file-meta">${file.chunks} Chunks · ${file.images} Images</span>
        </div>
      `;
      filesList.appendChild(item);
    });
  }

  // --- Messaging (Chat API Calls) ---
  async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    // Reset input height
    chatInput.value = '';
    chatInput.style.height = 'auto';
    sendBtn.disabled = true;

    // Append User message to window
    appendUserMessage(text);

    // Show Typing Indicator
    const typingIndicator = appendTypingIndicator();

    try {
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: text,
          thread_id: threadId,
          model_name: getSelectedModel()
        })
      });

      removeTypingIndicator(typingIndicator);

      if (!res.ok) {
        let errMsg = 'Failed to fetch response';
        try {
          const errData = await res.json();
          errMsg = errData.detail || errMsg;
        } catch (_) {}
        appendAIMessage(`Error: ${errMsg}`);
        sendBtn.disabled = false;
        return;
      }

      // Read SSE stream
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      
      const contentEl = appendStreamingPlaceholder();
      let answerText = '';
      let sources = [];

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        
        // Retain last incomplete line in buffer
        buffer = lines.pop();

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data: ')) continue;
          
          try {
            const data = JSON.parse(trimmed.substring(6));
            if (data.type === 'token') {
              answerText += data.text;
              contentEl.textContent = answerText;
              scrollToBottom();
            } else if (data.type === 'sources') {
              sources = data.sources;
            } else if (data.type === 'error') {
              answerText += `\n\n[System Error: ${data.detail}]`;
              contentEl.textContent = answerText;
            }
          } catch (e) {
            console.error('Failed to parse stream event:', e);
          }
        }
      }
      
      // Finalize markdown and source layout
      finalizeStreamingMessage(contentEl, answerText, sources);

    } catch (err) {
      removeTypingIndicator(typingIndicator);
      appendAIMessage(`Error: ${err.message}`);
    }
    
    sendBtn.disabled = false;
  }

  // --- UI Helpers for Chat Window ---
  function scrollToBottom() {
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }

  function appendUserMessage(text) {
    const row = document.createElement('div');
    row.className = 'message user-message';
    row.innerHTML = `
      <div class="avatar">U</div>
      <div class="message-content">
        <p>${escapeHtml(text)}</p>
      </div>
    `;
    chatWindow.appendChild(row);
    scrollToBottom();
  }

  function appendSystemMessage(text) {
    const row = document.createElement('div');
    row.className = 'message ai-message';
    row.innerHTML = `
      <div class="avatar">✦</div>
      <div class="message-content" style="font-style: italic; color: var(--text-muted);">
        <p>${escapeHtml(text)}</p>
      </div>
    `;
    chatWindow.appendChild(row);
    scrollToBottom();
  }

  function appendTypingIndicator() {
    const row = document.createElement('div');
    row.className = 'message ai-message';
    row.innerHTML = `
      <div class="avatar">✦</div>
      <div class="message-content">
        <div class="typing-dots">
          <span></span>
          <span></span>
          <span></span>
        </div>
      </div>
    `;
    chatWindow.appendChild(row);
    scrollToBottom();
    return row;
  }

  function removeTypingIndicator(el) {
    if (el) el.remove();
  }

  function renderAIMessageBody(contentEl, text, sources) {
    // Parse Markdown safely
    if (window.marked) {
      contentEl.innerHTML = marked.parse(text);
    } else {
      contentEl.textContent = text;
    }

    // Process sources if available
    if (sources && sources.length > 0) {
      const sourcesBlock = document.createElement('div');
      sourcesBlock.className = 'sources-block';
      sourcesBlock.innerHTML = `<div class="sources-title">Sources Used</div>`;

      const chipsContainer = document.createElement('div');
      chipsContainer.className = 'source-chips-container';

      const imagesGrid = document.createElement('div');
      imagesGrid.className = 'source-images-grid';

      let hasImages = false;

      sources.forEach(src => {
        // Create Chip
        const chip = document.createElement('span');
        chip.className = 'source-chip';
        const icon = src.type === 'image' ? '🖼️' : '📄';
        chip.innerHTML = `<span class="source-chip-icon">${icon}</span> ${src.source} (p. ${src.page})`;
        chipsContainer.appendChild(chip);

        // Create Image Preview
        if (src.type === 'image' && src.b64) {
          hasImages = true;
          const imgWrapper = document.createElement('div');
          imgWrapper.className = 'source-image-wrapper';
          imgWrapper.innerHTML = `
            <img src="data:${src.mime};base64,${src.b64}" alt="Source figure page ${src.page}" />
            <div class="image-caption">Page ${src.page}</div>
          `;
          
          // Image Modal Trigger
          imgWrapper.addEventListener('click', () => {
            showImageModal(src.b64, src.mime);
          });
          
          imagesGrid.appendChild(imgWrapper);
        }
      });

      sourcesBlock.appendChild(chipsContainer);
      if (hasImages) {
        sourcesBlock.appendChild(imagesGrid);
      }

      contentEl.appendChild(sourcesBlock);
    }
  }

  function appendAIMessage(text, sources) {
    const row = document.createElement('div');
    row.className = 'message ai-message';

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = '✦';

    const content = document.createElement('div');
    content.className = 'message-content';
    
    renderAIMessageBody(content, text, sources);

    row.appendChild(avatar);
    row.appendChild(content);
    chatWindow.appendChild(row);
    scrollToBottom();
  }

  function appendStreamingPlaceholder() {
    const row = document.createElement('div');
    row.className = 'message ai-message';

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = '✦';

    const content = document.createElement('div');
    content.className = 'message-content';
    content.textContent = '...';

    row.appendChild(avatar);
    row.appendChild(content);
    chatWindow.appendChild(row);
    scrollToBottom();
    
    return content;
  }

  function finalizeStreamingMessage(contentEl, text, sources) {
    contentEl.textContent = '';
    renderAIMessageBody(contentEl, text, sources);
    scrollToBottom();
  }

  // --- Image zoom Modal ---
  function showImageModal(b64, mime) {
    const modal = document.createElement('div');
    modal.className = 'modal';
    modal.innerHTML = `<img src="data:${mime};base64,${b64}" alt="Zoomed figure" />`;
    modal.addEventListener('click', () => modal.remove());
    document.body.appendChild(modal);
  }

  // --- Helper to escape HTML ---
  function escapeHtml(str) {
    return str.replace(/&/g, "&amp;")
              .replace(/</g, "&lt;")
              .replace(/>/g, "&gt;")
              .replace(/"/g, "&quot;")
              .replace(/'/g, "&#039;");
  }

  // Fire on load
  init();
})();

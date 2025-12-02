// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    await checkHealth();
    setupEventListeners();
});

function setupEventListeners() {
    const input = document.getElementById('question-input');
    input.addEventListener('input', updateWordCount);
    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) {
            submitQuery();
        }
    });
}

function updateWordCount() {
    const input = document.getElementById('question-input');
    const counter = document.getElementById('word-count');
    const words = input.value.trim().split(/\s+/).filter(w => w.length > 0);
    const count = words.length;
    
    counter.textContent = count;
    counter.parentElement.classList.remove('warning', 'error');
    
    if (count > 200) {
        counter.parentElement.classList.add('error');
    } else if (count > 180) {
        counter.parentElement.classList.add('warning');
    }
}

async function checkHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        
        document.getElementById('embedding-model').textContent = 
            data.embedding_model.split('/').pop();
        document.getElementById('retrieval-model').textContent = 
            data.retrieval_model;
        
        const status = document.getElementById('status');
        if (data.ollama_available) {
            status.classList.remove('error');
        } else {
            status.classList.add('error');
        }
    } catch (error) {
        console.error('Health check failed:', error);
        document.getElementById('status').classList.add('error');
    }
}

async function submitQuery() {
    const input = document.getElementById('question-input');
    const question = input.value.trim();
    
    if (!question) {
        alert('Please enter a question');
        return;
    }
    
    const words = question.split(/\s+/).filter(w => w.length > 0);
    if (words.length > 200) {
        alert('Question exceeds 200 words limit');
        return;
    }
    
    // Disable input
    setLoading(true);
    
    // Add user message
    addMessage('user', question);
    
    try {
        const response = await fetch('/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: question,
                top_k: 5,
            }),
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        addMessage('assistant', data.answer, data.sources);
        
        // Clear input
        input.value = '';
        updateWordCount();
        
    } catch (error) {
        console.error('Query failed:', error);
        addMessage('system', `Error: ${error.message}`);
    } finally {
        setLoading(false);
    }
}

function setLoading(loading) {
    const btn = document.getElementById('submit-btn');
    const input = document.getElementById('question-input');
    const btnText = document.getElementById('btn-text');
    const btnLoader = document.getElementById('btn-loader');
    
    btn.disabled = loading;
    input.disabled = loading;
    
    if (loading) {
        btnText.style.display = 'none';
        btnLoader.style.display = 'inline-block';
    } else {
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
    }
}

function addMessage(type, content, sources = null) {
    const chatHistory = document.getElementById('chat-history');
    const messageDiv = document.createElement('div');
    
    if (type === 'user') {
        messageDiv.className = 'user-message';
        messageDiv.innerHTML = `<strong>You:</strong> ${escapeHtml(content)}`;
    } else if (type === 'assistant') {
        messageDiv.className = 'assistant-message';
        let html = `<div class="answer-text">${escapeHtml(content)}</div>`;
        
        if (sources && sources.length > 0) {
            html += '<div class="sources"><strong>Sources:</strong>';
            sources.forEach((src, idx) => {
                html += `<div class="source-item">
                    [${idx + 1}] ${escapeHtml(src.snippet)} 
                    <em>(Score: ${src.score.toFixed(3)})</em>
                </div>`;
            });
            html += '</div>';
        }
        
        messageDiv.innerHTML = html;
    } else {
        messageDiv.className = 'system-message';
        messageDiv.textContent = content;
    }
    
    chatHistory.appendChild(messageDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

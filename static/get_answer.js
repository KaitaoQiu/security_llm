let history = [];

async function getAnswer() {
    const question = document.getElementById('questionInput').value;
    if (!question) return; 

    const response = await fetch('/get-answer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: question }),
    });
    const data = await response.json();
    const answer = data.answer;

    history.push({ question, answer });

    updateHistoryUI();
}

function updateHistoryUI() {
    const historyElement = document.getElementById('history');
    historyElement.innerHTML = '';

    history.forEach(item => {
        const entry = document.createElement('div');
        // entry.innerHTML = `<strong>Q:</strong> ${item.question} <br> <strong>A:</strong> ${item.answer}`;
        entry.innerHTML = `
        <div style="margin-bottom: 20px; text-decoration: none;">
            <strong>Q:</strong> ${item.question} <br>
            <strong>A:</strong> ${item.answer}
        </div>
        `;
        historyElement.appendChild(entry);
    });
}

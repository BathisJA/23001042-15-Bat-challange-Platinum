async function submitTextForm(inputId, modelType) {
    const inputValue = document.getElementById(inputId).value;

    if (inputValue.trim() !== '') {
        const formData = new FormData();
        formData.append('text', inputValue);
    
        try {
            const url = modelType === 'NN' ?
                'http://127.0.0.1:5000/text-processing_NN' :
                'http://127.0.0.1:5000/text-processing_LSTM';

            const response = await fetch(url, {
                method: 'POST',
                body: formData,
            });

            const responseData = await response.json();
            if (modelType === 'NN') {
                displayOutput(responseData, inputId);
            } else {
                displayOutput(responseData, inputId); 
            }
        } catch (error) {
            console.error('Error:', error.message);
            displayError('Failed to process the text.');
        }
    } else {
        alert('Input cannot be empty.');
    }
}


async function submitFileForm(inputId, modelType) {
    const fileInput = document.getElementById(inputId);
    const file = fileInput.files[0];

    if (file) {
        const formData = new FormData();
        formData.append('file', file);
    
        try {
            const url = modelType === 'NN' ?
                'http://127.0.0.1:5000/text-processing-file_NN' :
                'http://127.0.0.1:5000/text-processing-file_LSTM';
        
            const response = await fetch(url, {
                method: 'POST',
                body: formData,
            });
        
            const responseData = await response.json();
            if (modelType === 'NN') {
                displayOutputFile(responseData, inputId);
            } else {
                displayOutputFile(responseData, inputId);
            }
        } catch (error) {
            console.error('Error:', error.message);
            displayError('Failed to process the file.');
        }
    } else {
        alert('Please select a file.');
    }
}

function displayOutput(responseData, inputId) {
    const tableBody = document.getElementById('outputBody');
    const newRow = tableBody.insertRow(0);
    const newCellText = newRow.insertCell();
    const newCellSentiment = newRow.insertCell();

    newCellText.textContent = responseData.data.text;
    newCellSentiment.textContent = responseData.data.sentiment;

    document.getElementById('responseTitle').textContent = 'API Response:';
    document.getElementById('responseStatus').textContent = `Status: ${responseData.status_code}`;
    document.getElementById('outputTable').style.display = 'table';
    document.querySelector('.reset-button').style.display = 'block';

    document.getElementById(inputId).value = '';
}

function displayOutputFile(responseData, inputId) {
    const tableBody = document.getElementById('outputBody');

    if (Array.isArray(responseData.data)) {
        responseData.data.forEach(data => {
            const newRow = tableBody.insertRow(0);
            const newCellText = newRow.insertCell();
            const newCellSentiment = newRow.insertCell();

            newCellText.textContent = data.text;
            newCellSentiment.textContent = data.sentimen;
        });

        document.getElementById('responseTitle').textContent = 'API Response:';
        document.getElementById('responseStatus').textContent = `Status: ${responseData.status_code}`;
        document.getElementById('outputTable').style.display = 'table';
        document.querySelector('.reset-button').style.display = 'block';
    
        document.getElementById(inputId).value = '';
    }
}

function displayError(status) {
    updateResponseDisplay('API Error:', status);
    resetOutputForm('');
}


function displayError(status) {
    updateResponseDisplay('API Error:', status);
    resetOutputForm('');
}

function updateResponseDisplay(title, status) {
    document.getElementById('responseTitle').textContent = title;
    document.getElementById('responseStatus').textContent = `Status: ${status}`;
    document.getElementById('outputTable').style.display = 'table';
    document.querySelector('.reset-button').style.display = 'block';
}

function resetOutputForm(inputId) {
    document.getElementById('outputBody').innerHTML = '';
    document.getElementById(inputId).value = '';
}

function resetOutput() {
    const tableBody = document.getElementById('outputBody');
    tableBody.innerHTML = '';

    document.getElementById('outputTable').style.display = 'none';
    document.querySelector('.reset-button').style.display = 'none';
    document.getElementById('responseTitle').textContent = '';
    document.getElementById('responseStatus').textContent = '';

    document.querySelectorAll('input[type="text"]').forEach(input => {
        input.value = '';
    });

    document.querySelectorAll('input[type="file"]').forEach(input => {
        input.value = '';
    });
}
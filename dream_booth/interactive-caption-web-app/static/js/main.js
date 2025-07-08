document.addEventListener('DOMContentLoaded', function() {
    const saveButton = document.getElementById('save-caption');
    const saveAllButton = document.getElementById('save-all');
    const generateButton = document.getElementById('generate-caption');
    const captionForm = document.getElementById('caption-form');
    const captionTextarea = document.getElementById('caption');
    const messageDiv = document.getElementById('message');

    // 表情からキャプション生成
    if (generateButton) {
        generateButton.addEventListener('click', function() {
            const expression = document.getElementById('expression').value;
            const intensity = document.getElementById('intensity').value;
            
            if (!expression || !intensity) {
                showMessage('表情と強度を両方選択してください', 'error');
                return;
            }
            
            const formData = new FormData();
            formData.append('expression', expression);
            formData.append('intensity', intensity);
            
            fetch('/generate_caption', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    captionTextarea.value = data.caption;
                    showMessage('キャプションが生成されました', 'success');
                } else {
                    showMessage(data.message || '生成に失敗しました', 'error');
                }
            })
            .catch(error => {
                showMessage('エラーが発生しました', 'error');
            });
        });
    }

    // キャプション保存
    if (saveButton) {
        saveButton.addEventListener('click', function() {
            const formData = new FormData(captionForm);
            
            fetch('/save_caption', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showMessage('キャプションが保存されました', 'success');
                } else {
                    showMessage('保存に失敗しました', 'error');
                }
            })
            .catch(error => {
                showMessage('エラーが発生しました', 'error');
            });
        });
    }

    // 全て保存
    if (saveAllButton) {
        saveAllButton.addEventListener('click', function() {
            fetch('/save_all', {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showMessage(`${data.saved_count}件のキャプションが保存されました`, 'success');
                } else {
                    showMessage(data.message || '保存に失敗しました', 'error');
                }
            })
            .catch(error => {
                showMessage('エラーが発生しました', 'error');
            });
        });
    }

    // キーボードショートカット
    document.addEventListener('keydown', function(e) {
        if (e.ctrlKey && e.key === 's') {
            e.preventDefault();
            if (saveButton) {
                saveButton.click();
            }
        }
        if (e.ctrlKey && e.key === 'g') {
            e.preventDefault();
            if (generateButton) {
                generateButton.click();
            }
        }
    });

    function showMessage(text, type) {
        messageDiv.textContent = text;
        messageDiv.className = `message ${type}`;
        messageDiv.style.display = 'block';
        
        setTimeout(function() {
            messageDiv.style.display = 'none';
        }, 3000);
    }
});
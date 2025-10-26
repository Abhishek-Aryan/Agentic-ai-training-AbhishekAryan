// Random Forest Analysis Tool - Main JavaScript

document.addEventListener('DOMContentLoaded', function() {
    initializeFileUpload();
    initializeFormValidation();
    initializeAnimations();
});

// File Upload Handling
function initializeFileUpload() {
    const fileInput = document.getElementById('fileInput');
    const fileUploadArea = document.getElementById('fileUploadArea');
    
    if (fileInput && fileUploadArea) {
        // Drag and drop functionality
        ['dragover', 'dragenter'].forEach(eventName => {
            fileUploadArea.addEventListener(eventName, function(e) {
                e.preventDefault();
                this.classList.add('dragover');
            });
        });
        
        ['dragleave', 'dragend'].forEach(eventName => {
            fileUploadArea.addEventListener(eventName, function(e) {
                e.preventDefault();
                this.classList.remove('dragover');
            });
        });
        
        fileUploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            this.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFileSelection(files[0]);
            }
        });
        
        // File input change
        fileInput.addEventListener('change', function(e) {
            if (this.files.length > 0) {
                handleFileSelection(this.files[0]);
            }
        });
        
        function handleFileSelection(file) {
            if (!isValidFile(file)) {
                showError('Please select a CSV or Excel file (max 16MB)');
                return;
            }
            
            updateFileDisplay(file);
            showSuccess('File selected successfully: ' + file.name);
        }
        
        function isValidFile(file) {
            const allowedTypes = ['.csv', '.xlsx', '.xls'];
            const maxSize = 16 * 1024 * 1024; // 16MB
            
            const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
            return allowedTypes.includes(fileExtension) && file.size <= maxSize;
        }
        
        function updateFileDisplay(file) {
            const fileName = file.name;
            const fileSize = (file.size / (1024 * 1024)).toFixed(2);
            
            fileUploadArea.innerHTML = `
                <i class="fas fa-file-check" style="color: #28a745;"></i>
                <h3>${fileName}</h3>
                <p>Size: ${fileSize} MB</p>
                <p style="color: #28a745;"><i class="fas fa-check-circle"></i> Ready for analysis</p>
                <label for="fileInput" class="file-label">Change File</label>
            `;
            
            // Re-attach the file input
            const newFileInput = document.createElement('input');
            newFileInput.type = 'file';
            newFileInput.name = 'file';
            newFileInput.id = 'fileInput';
            newFileInput.accept = '.csv,.xlsx,.xls';
            newFileInput.style.display = 'none';
            newFileInput.addEventListener('change', function(e) {
                if (this.files.length > 0) {
                    handleFileSelection(this.files[0]);
                }
            });
            
            fileUploadArea.appendChild(newFileInput);
        }
    }
}

// Form Validation
function initializeFormValidation() {
    const forms = document.querySelectorAll('form');
    
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            if (!validateForm(this)) {
                e.preventDefault();
                showError('Please fix the errors in the form before submitting.');
            }
        });
        
        // Real-time validation
        const inputs = form.querySelectorAll('input[required], select[required]');
        inputs.forEach(input => {
            input.addEventListener('blur', function() {
                validateField(this);
            });
            
            input.addEventListener('input', function() {
                clearFieldError(this);
            });
        });
    });
}

function validateForm(form) {
    const requiredFields = form.querySelectorAll('[required]');
    let isValid = true;
    
    requiredFields.forEach(field => {
        if (!validateField(field)) {
            isValid = false;
        }
    });
    
    return isValid;
}

function validateField(field) {
    const value = field.value.trim();
    let isValid = true;
    let errorMessage = '';
    
    // Clear previous error
    clearFieldError(field);
    
    // Check required fields
    if (field.hasAttribute('required') && !value) {
        isValid = false;
        errorMessage = 'This field is required';
    }
    
    // Check numeric fields
    if (field.type === 'number' && value) {
        const min = parseFloat(field.getAttribute('min'));
        const max = parseFloat(field.getAttribute('max'));
        const numValue = parseFloat(value);
        
        if (!isNaN(min) && numValue < min) {
            isValid = false;
            errorMessage = `Value must be at least ${min}`;
        }
        
        if (!isNaN(max) && numValue > max) {
            isValid = false;
            errorMessage = `Value must be at most ${max}`;
        }
    }
    
    if (!isValid) {
        showFieldError(field, errorMessage);
    } else {
        showFieldSuccess(field);
    }
    
    return isValid;
}

function showFieldError(field, message) {
    field.style.borderColor = '#dc3545';
    field.style.boxShadow = '0 0 0 0.2rem rgba(220, 53, 69, 0.25)';
    
    // Remove existing error message
    const existingError = field.parentNode.querySelector('.field-error');
    if (existingError) {
        existingError.remove();
    }
    
    // Add error message
    const errorElement = document.createElement('div');
    errorElement.className = 'field-error';
    errorElement.style.color = '#dc3545';
    errorElement.style.fontSize = '0.875rem';
    errorElement.style.marginTop = '0.25rem';
    errorElement.innerHTML = `<i class="fas fa-exclamation-circle"></i> ${message}`;
    
    field.parentNode.appendChild(errorElement);
}

function showFieldSuccess(field) {
    field.style.borderColor = '#28a745';
    field.style.boxShadow = '0 0 0 0.2rem rgba(40, 167, 69, 0.25)';
    
    // Remove existing error message
    const existingError = field.parentNode.querySelector('.field-error');
    if (existingError) {
        existingError.remove();
    }
}

function clearFieldError(field) {
    field.style.borderColor = '';
    field.style.boxShadow = '';
    
    const existingError = field.parentNode.querySelector('.field-error');
    if (existingError) {
        existingError.remove();
    }
}

// Animations
function initializeAnimations() {
    // Animate metric cards on results page
    if (document.querySelector('.results-container')) {
        animateMetricCards();
    }
    
    // Add scroll animations
    const animatedElements = document.querySelectorAll('.feature-card, .step, .feature-item');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, { threshold: 0.1 });
    
    animatedElements.forEach(element => {
        element.style.opacity = '0';
        element.style.transform = 'translateY(30px)';
        element.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(element);
    });
}

function animateMetricCards() {
    const metricCards = document.querySelectorAll('.metric-card');
    
    metricCards.forEach((card, index) => {
        card.style.animationDelay = `${index * 0.1}s`;
    });
}

// Utility Functions
function showError(message) {
    showNotification(message, 'error');
}

function showSuccess(message) {
    showNotification(message, 'success');
}

function showNotification(message, type) {
    // Remove existing notifications
    const existingNotifications = document.querySelectorAll('.custom-notification');
    existingNotifications.forEach(notification => notification.remove());
    
    const notification = document.createElement('div');
    notification.className = `custom-notification ${type}`;
    notification.innerHTML = `
        <i class="fas fa-${type === 'error' ? 'exclamation-circle' : 'check-circle'}"></i>
        ${message}
    `;
    
    // Add styles
    notification.style.cssText = `
        position: fixed;
        top: 100px;
        right: 20px;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        z-index: 10000;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        display: flex;
        align-items: center;
        gap: 0.5rem;
        max-width: 400px;
        animation: slideInRight 0.3s ease;
    `;
    
    if (type === 'error') {
        notification.style.background = 'linear-gradient(135deg, #dc3545, #c82333)';
    } else {
        notification.style.background = 'linear-gradient(135deg, #28a745, #1e7e34)';
    }
    
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.style.animation = 'slideOutRight 0.3s ease';
            setTimeout(() => notification.remove(), 300);
        }
    }, 5000);
}

// Add CSS for animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
    
    .progress-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.7);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 9999;
        color: white;
    }
    
    .progress-content {
        text-align: center;
        background: white;
        padding: 2rem;
        border-radius: 12px;
        color: #333;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    
    .spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #3498db;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 2s linear infinite;
        margin: 0 auto 1rem;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
`;
document.head.appendChild(style);

// Progress indicator for long-running operations
function showProgress(message = 'Processing your analysis...') {
    const progressDiv = document.createElement('div');
    progressDiv.className = 'progress-overlay';
    progressDiv.innerHTML = `
        <div class="progress-content">
            <div class="spinner"></div>
            <h3>Random Forest Analysis</h3>
            <p>${message}</p>
            <p><small>This may take a few moments depending on your dataset size</small></p>
        </div>
    `;
    
    document.body.appendChild(progressDiv);
    return progressDiv;
}

function hideProgress(progressDiv) {
    if (progressDiv && progressDiv.parentNode) {
        progressDiv.parentNode.removeChild(progressDiv);
    }
}

// Export functions for global use
window.showProgress = showProgress;
window.hideProgress = hideProgress;
window.showError = showError;
window.showSuccess = showSuccess;
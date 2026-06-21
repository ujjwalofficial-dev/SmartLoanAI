// ============================================
// SPLASH SCREEN
// ============================================

(function () {
    const container = document.getElementById('splashParticles');
    if (container) {
        for (let i = 0; i < 35; i++) {
            const p = document.createElement('div');
            p.className = 'splash-particle';
            const size = Math.random() * 6 + 3;
            p.style.cssText = [
                'width:'  + size + 'px',
                'height:' + size + 'px',
                'left:'   + Math.random() * 100 + '%',
                'bottom:' + (Math.random() * 20 - 10) + '%',
                'animation-duration:' + (Math.random() * 8 + 6) + 's',
                'animation-delay:'    + (Math.random() * 5) + 's',
                'opacity:' + (Math.random() * 0.5 + 0.1)
            ].join(';');
            container.appendChild(p);
        }
    }

    const btn     = document.getElementById('getStartedBtn');
    const splash  = document.getElementById('splashScreen');
    const mainApp = document.getElementById('mainApp');

    if (btn && splash && mainApp) {
        btn.addEventListener('click', function () {
            splash.classList.add('fade-out');
            setTimeout(function () {
                splash.style.display = 'none';
                mainApp.classList.add('visible');
            }, 800);
        });
    }
})();

// ============================================
// CHART INSTANCE
// ============================================

let approvalChart = null;

// ============================================
// TAB SWITCHING
// ============================================

document.querySelectorAll('.tab-button').forEach(button => {
    button.addEventListener('click', function() {
        const tabName = this.getAttribute('data-tab');

        document.querySelectorAll('.tab-content').forEach(tab => {
            tab.classList.remove('active');
        });

        document.querySelectorAll('.tab-button').forEach(btn => {
            btn.classList.remove('active');
        });

        document.getElementById(tabName).classList.add('active');
        this.classList.add('active');

        if (tabName === 'weights') {
            loadFeatureWeights();
        }
    });
});

// ============================================
// LOAD FEATURE WEIGHTS
// ============================================

function loadFeatureWeights() {
    fetch('/api/feature_weights')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                const weightsList = document.getElementById('weightsList');
                weightsList.innerHTML = '';

                data.weights.forEach((item) => {
                    const weightItem = document.createElement('div');
                    weightItem.className = 'weight-item';
                    weightItem.innerHTML = `
                        <div class="weight-header">
                            <span class="weight-name">${item.feature.replace(/_/g, ' ')}</span>
                            <span class="weight-percentage">${item.percentage}%</span>
                        </div>
                        <div class="weight-bar">
                            <div class="weight-fill" style="width: ${item.percentage}%"></div>
                        </div>
                    `;
                    weightsList.appendChild(weightItem);
                });
            }
        })
        .catch(error => {
            console.error('Error loading weights:', error);
            alert('Error loading feature weights');
        });
}

// ============================================
// FORM SUBMISSION
// ============================================

document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();

    const formData = {
        gender:             document.getElementById('gender').value,
        married:            document.getElementById('married').value,
        dependents:         document.getElementById('dependents').value,
        education:          document.getElementById('education').value,
        self_employed:      document.getElementById('self_employed').value,
        property_area:      document.getElementById('property_area').value,
        applicant_income:   document.getElementById('applicant_income').value,
        coapplicant_income: document.getElementById('coapplicant_income').value,
        loan_amount:        document.getElementById('loan_amount').value,
        loan_term:          document.getElementById('loan_term').value,
        credit_history:     document.getElementById('credit_history').value
    };

    for (let key in formData) {
        if (!formData[key]) {
            alert(`Please fill in all fields. Missing: ${key}`);
            return;
        }
    }

    try {
        const submitBtn    = document.querySelector('.submit-btn');
        const originalText = submitBtn.innerHTML;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        submitBtn.disabled  = true;

        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });

        const result = await response.json();

        if (result.status === 'success') {
            displayResults(result, formData);
            setTimeout(() => {
                document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
            }, 100);
        } else {
            alert('Prediction failed: ' + result.message);
        }

        submitBtn.innerHTML = originalText;
        submitBtn.disabled  = false;

    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred: ' + error.message);
        const submitBtn = document.querySelector('.submit-btn');
        submitBtn.innerHTML = '<i class="fas fa-play-circle"></i> Check Loan Approval';
        submitBtn.disabled  = false;
    }
});

// ============================================
// DISPLAY RESULTS
// ============================================

function displayResults(result, formData) {
    document.getElementById('resultsSection').classList.remove('hidden');

    const approvalProb = (result.approval_probability * 100).toFixed(2);
    document.getElementById('approvalProb').textContent    = approvalProb + '%';
    document.getElementById('progressFill').style.width    = approvalProb + '%';
    document.getElementById('eligibilityScore').textContent = result.eligibility_score + '/100';

    const riskBadge = document.getElementById('riskBadge');
    riskBadge.textContent = result.risk_level;
    riskBadge.className   = 'risk-badge ' + result.risk_color;

    updateApprovalChart(result.approval_chance, result.rejection_chance);

    const totalIncome = parseFloat(formData.applicant_income) + parseFloat(formData.coapplicant_income);
    const loanAmount  = parseFloat(formData.loan_amount);
    const ltiRatio    = ((loanAmount / totalIncome) * 100).toFixed(2);

    document.getElementById('totalIncomeDisplay').textContent = '₹' + totalIncome.toLocaleString('en-IN');
    document.getElementById('ltiRatioDisplay').textContent    = ltiRatio + '%';

    let recommendation = '';
    if (result.approval_probability >= 0.80) {
        recommendation = '✓ Excellent chance of approval! Your profile is strong.';
    } else if (result.approval_probability >= 0.60) {
        recommendation = '→ Good chance of approval. Review and improve credit profile if needed.';
    } else if (result.approval_probability >= 0.40) {
        recommendation = '⚠ Moderate chance. Consider improving credit history or reducing loan amount.';
    } else {
        recommendation = '✗ Low chance of approval. Improve credit score or increase income before applying.';
    }

    document.getElementById('recommendationDisplay').textContent = recommendation;
}

// ============================================
// UPDATE CHART
// ============================================

function updateApprovalChart(approvalChance, rejectionChance) {
    const ctx = document.getElementById('approvalChart').getContext('2d');

    if (approvalChart) {
        approvalChart.destroy();
    }

    approvalChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Approval Chance', 'Rejection Chance'],
            datasets: [{
                data: [approvalChance, rejectionChance],
                backgroundColor: [
                    'rgba(56, 189, 248, 0.7)',
                    'rgba(239, 68, 68, 0.7)'
                ],
                borderColor: [
                    'rgba(56, 189, 248, 1)',
                    'rgba(239, 68, 68, 1)'
                ],
                borderWidth: 2,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            animation: { duration: 1500, easing: 'easeInOutQuart' },
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#cbd5e1',
                        padding: 20,
                        font: { size: 14, weight: 'bold' }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.label + ': ' + context.parsed + '%';
                        }
                    },
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleColor: '#38bdf8',
                    bodyColor: '#cbd5e1',
                    borderColor: 'rgba(56, 189, 248, 0.5)',
                    borderWidth: 1,
                    padding: 10
                }
            }
        }
    });
}

// ============================================
// PAGE LOAD
// ============================================

document.addEventListener('DOMContentLoaded', function() {
    console.log('Finance MITRA - Ready');

    fetch('/api/health')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                console.log('✓ API Connected');
            }
        })
        .catch(error => console.error('API Health Check Failed:', error));
});

// ============================================
// SMOOTH SCROLL
// ============================================

document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({ behavior: 'smooth' });
        }
    });
});

// ============================================
// DISABLE SPECIAL CHARACTERS IN NUMERIC INPUTS
// ============================================

document.querySelectorAll('input[type="number"]').forEach(input => {
    input.addEventListener('keypress', function(e) {
        if (!/[\d]/.test(e.key) && !['Backspace', 'ArrowLeft', 'ArrowRight', 'Delete'].includes(e.key)) {
            e.preventDefault();
        }
    });
});
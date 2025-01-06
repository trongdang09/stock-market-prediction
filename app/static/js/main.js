// Khởi tạo biểu đồ
function initCharts() {
    // Biểu đồ VN-Index
    const vnIndexTrace = {
        x: [],
        y: [],
        type: 'scatter',
        name: 'VN-Index'
    };

    const vnIndexLayout = {
        title: 'VN-Index Prediction',
        xaxis: { title: 'Date' },
        yaxis: { title: 'Price' }
    };

    Plotly.newPlot('vn-index-chart', [vnIndexTrace], vnIndexLayout);

    // Biểu đồ S&P 500
    const sp500Trace = {
        x: [],
        y: [],
        type: 'scatter',
        name: 'S&P 500'
    };

    const sp500Layout = {
        title: 'S&P 500 Prediction',
        xaxis: { title: 'Date' },
        yaxis: { title: 'Price' }
    };

    Plotly.newPlot('sp500-chart', [sp500Trace], sp500Layout);
}

// Phân tích cổ phiếu
async function analyzeStock() {
    const symbol = document.getElementById('stockSymbol').value;
    if (!symbol) {
        alert('Vui lòng nhập mã cổ phiếu');
        return;
    }

    try {
        const response = await fetch(`/api/analyze/${symbol}`);
        const data = await response.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        // Cập nhật biểu đồ
        updatePriceChart(data.prices);
        
        // Cập nhật chỉ số kỹ thuật
        document.getElementById('rsi-value').textContent = data.indicators.rsi;
        document.getElementById('macd-value').textContent = data.indicators.macd;
        document.getElementById('bb-value').textContent = data.indicators.bollinger;

        // Cập nhật tin tức
        updateMarketNews(data.news);

    } catch (error) {
        console.error('Error:', error);
        alert('Có lỗi xảy ra khi phân tích cổ phiếu');
    }
}

// Lấy dự đoán AI
async function getPrediction() {
    const symbol = document.getElementById('predictionSymbol').value;
    const modelType = document.getElementById('modelType').value;

    if (!symbol) {
        alert('Vui lòng nhập mã cổ phiếu');
        return;
    }

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                symbol: symbol,
                model: modelType
            })
        });

        const data = await response.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        // Cập nhật kết quả dự đoán
        document.getElementById('pred-1d').textContent = data.predictions.oneDay;
        document.getElementById('pred-1w').textContent = data.predictions.oneWeek;
        document.getElementById('pred-1m').textContent = data.predictions.oneMonth;
        document.getElementById('confidence').textContent = data.confidence + '%';

        // Cập nhật biểu đồ dự đoán
        updatePredictionChart(data.chartData);

        // Cập nhật phân tích rủi ro
        updateRiskAnalysis(data.risk);

    } catch (error) {
        console.error('Error:', error);
        alert('Có lỗi xảy ra khi dự đoán');
    }
}

// Cập nhật biểu đồ giá
function updatePriceChart(data) {
    const trace = {
        x: data.dates,
        y: data.prices,
        type: 'scatter',
        name: 'Price'
    };

    const layout = {
        title: 'Stock Price History',
        xaxis: { title: 'Date' },
        yaxis: { title: 'Price' }
    };

    Plotly.newPlot('price-chart', [trace], layout);
}

// Cập nhật tin tức thị trường
function updateMarketNews(news) {
    const newsContainer = document.getElementById('market-news');
    newsContainer.innerHTML = '';

    news.forEach(item => {
        const newsItem = document.createElement('div');
        newsItem.className = 'news-item';
        newsItem.innerHTML = `
            <h5>${item.title}</h5>
            <p>${item.summary}</p>
            <small>${item.date}</small>
        `;
        newsContainer.appendChild(newsItem);
    });
}

// Cập nhật biểu đồ dự đoán
function updatePredictionChart(data) {
    const historicalTrace = {
        x: data.dates,
        y: data.historical,
        type: 'scatter',
        name: 'Historical'
    };

    const predictionTrace = {
        x: data.predDates,
        y: data.predictions,
        type: 'scatter',
        name: 'Prediction',
        line: {
            dash: 'dot',
            color: 'red'
        }
    };

    const layout = {
        title: 'Price Prediction',
        xaxis: { title: 'Date' },
        yaxis: { title: 'Price' }
    };

    Plotly.newPlot('prediction-chart', [historicalTrace, predictionTrace], layout);
}

// Cập nhật phân tích rủi ro
function updateRiskAnalysis(risk) {
    const riskLevel = document.getElementById('risk-level');
    const riskText = document.getElementById('risk-text');

    riskLevel.style.width = risk.level + '%';
    riskLevel.className = `progress-bar bg-${risk.level < 30 ? 'success' : risk.level < 70 ? 'warning' : 'danger'}`;
    riskText.textContent = risk.description;
}

// Khởi tạo khi trang được load
document.addEventListener('DOMContentLoaded', function() {
    initCharts();
}); 
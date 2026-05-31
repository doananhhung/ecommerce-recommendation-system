from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn

from src.config import config
from src.serving.pipeline import RecommendationPipeline

app = FastAPI(title="Two-Stage Recommender API", version="1.0.0")

# Global pipeline instance
pipeline = None

@app.on_event("startup")
async def startup_event():
    global pipeline
    
    pipeline = RecommendationPipeline(models_dir=config.MODELS_DIR, feature_store_dir=config.FEATURE_STORE_DIR)
    pipeline.initialize()
    print("API Server is ready to receive requests.")

@app.get("/", response_class=HTMLResponse)
def read_root():
    html_content = """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Two-Stage Recommender Dashboard</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary: #4F46E5;
                --primary-hover: #4338CA;
                --bg: #F3F4F6;
                --card-bg: #FFFFFF;
                --text-main: #111827;
                --text-muted: #6B7280;
                --border: #E5E7EB;
            }
            * { box-sizing: border-box; }
            body {
                font-family: 'Inter', sans-serif;
                background-color: var(--bg);
                color: var(--text-main);
                margin: 0;
                padding: 40px 20px;
                display: flex;
                justify-content: center;
                min-height: 100vh;
            }
            .container {
                background-color: var(--card-bg);
                padding: 40px;
                border-radius: 20px;
                box-shadow: 0 20px 25px -5px rgba(0,0,0,0.1), 0 10px 10px -5px rgba(0,0,0,0.04);
                max-width: 800px;
                width: 100%;
                margin-top: 20px;
                margin-bottom: 20px;
            }
            h1 {
                margin-top: 0;
                font-size: 28px;
                color: var(--text-main);
                display: flex;
                align-items: center;
                gap: 10px;
            }
            p { color: var(--text-muted); line-height: 1.6; }
            .form-group {
                display: flex;
                gap: 15px;
                margin-top: 30px;
                flex-wrap: wrap;
            }
            .input-wrapper {
                flex: 1;
                min-width: 200px;
                display: flex;
                flex-direction: column;
                gap: 8px;
            }
            label {
                font-weight: 600;
                font-size: 14px;
            }
            input {
                padding: 12px 16px;
                border: 1px solid var(--border);
                border-radius: 10px;
                font-size: 16px;
                font-family: inherit;
                transition: all 0.3s ease;
                outline: none;
            }
            input:focus {
                border-color: var(--primary);
                box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.2);
            }
            button {
                background-color: var(--primary);
                color: white;
                border: none;
                border-radius: 10px;
                padding: 12px 24px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: background-color 0.3s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                align-self: flex-end;
                height: 45px;
            }
            button:hover {
                background-color: var(--primary-hover);
            }
            button:disabled {
                background-color: var(--text-muted);
                cursor: not-allowed;
            }
            .examples {
                margin-top: 15px;
                font-size: 13px;
                color: var(--text-muted);
                background: #f9fafb;
                padding: 10px 15px;
                border-radius: 8px;
                border: 1px dashed var(--border);
            }
            #results-container {
                margin-top: 40px;
                display: none;
            }
            .results-header {
                font-size: 20px;
                font-weight: 700;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 2px solid var(--border);
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                gap: 15px;
            }
            .product-card {
                background: #ffffff;
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 15px;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            .product-card:hover {
                transform: translateY(-3px);
                box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
                border-color: var(--primary);
            }
            .product-id {
                font-weight: 700;
                font-size: 16px;
                color: var(--primary);
                margin-bottom: 5px;
                word-wrap: break-word;
            }
            .product-score {
                font-size: 13px;
                color: var(--text-muted);
                display: flex;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: 5px;
            }
            .loader {
                display: none;
                width: 20px;
                height: 20px;
                border: 3px solid rgba(255,255,255,0.3);
                border-radius: 50%;
                border-top-color: white;
                animation: spin 1s ease-in-out infinite;
            }
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
            .error-message {
                color: #DC2626;
                background: #FEE2E2;
                padding: 15px;
                border-radius: 10px;
                margin-top: 20px;
                display: none;
            }
            .json-raw {
                background: #1f2937;
                color: #e5e7eb;
                padding: 15px;
                border-radius: 10px;
                overflow-x: auto;
                font-family: monospace;
                font-size: 13px;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>✨ AI Recommender System</h1>
            <p>Hệ thống Gợi ý Sản phẩm 2 giai đoạn (Two-Stage Recommender). Hãy nhập ID của khách hàng để nhận danh sách sản phẩm gợi ý phù hợp nhất.</p>
            
            <div class="examples">
                💡 <b>Gợi ý một số User ID đã được training:</b> 315805600, 332550649, 343069863, 351866718, 353733558
            </div>

            <form id="recommend-form" class="form-group">
                <div class="input-wrapper">
                    <label for="user_id">User ID</label>
                    <input type="number" id="user_id" placeholder="VD: 244951053" required>
                </div>
                <div class="input-wrapper" style="flex: 0.5;">
                    <label for="top_n">Số lượng (Top N)</label>
                    <input type="number" id="top_n" value="20" min="1" max="100">
                </div>
                <button type="submit" id="submit-btn">
                    <span id="btn-text">Lấy Gợi ý</span>
                    <div class="loader" id="loader"></div>
                </button>
            </form>

            <div id="error-box" class="error-message"></div>

            <div id="results-container">
                <div class="results-header" id="results-title">Kết quả gợi ý</div>
                <div class="grid" id="products-grid"></div>
                <!-- Hiển thị raw data phòng trường hợp không parse được mảng items -->
                <div id="raw-data-container" style="display: none;">
                    <h4>Raw JSON Response:</h4>
                    <pre class="json-raw" id="raw-json"></pre>
                </div>
            </div>
        </div>

        <script>
            document.getElementById('recommend-form').addEventListener('submit', async (e) => {
                e.preventDefault();
                const userId = document.getElementById('user_id').value;
                const topN = document.getElementById('top_n').value || 20;
                const btnText = document.getElementById('btn-text');
                const loader = document.getElementById('loader');
                const btn = document.getElementById('submit-btn');
                const resultsContainer = document.getElementById('results-container');
                const productsGrid = document.getElementById('products-grid');
                const errorBox = document.getElementById('error-box');
                const rawDataContainer = document.getElementById('raw-data-container');
                const rawJson = document.getElementById('raw-json');

                // Reset UI
                errorBox.style.display = 'none';
                resultsContainer.style.display = 'none';
                productsGrid.innerHTML = '';
                rawDataContainer.style.display = 'none';
                btnText.style.display = 'none';
                loader.style.display = 'block';
                btn.disabled = true;

                try {
                    const response = await fetch(`/recommend/${userId}?top_n=${topN}`);
                    const data = await response.json();

                    if (!response.ok) {
                        throw new Error(data.detail || data.error || 'Có lỗi xảy ra từ server');
                    }

                    resultsContainer.style.display = 'block';
                    document.getElementById('results-title').innerText = `Kết quả gợi ý cho User: ${userId}`;

                    // Extract items gracefully
                    let items = [];
                    if (Array.isArray(data)) items = data;
                    else if (data.recommendations && Array.isArray(data.recommendations)) items = data.recommendations;
                    else if (data.items && Array.isArray(data.items)) items = data.items;
                    else if (data.result && Array.isArray(data.result)) items = data.result;

                    if (items.length > 0) {
                        items.forEach((item, index) => {
                            const card = document.createElement('div');
                            card.className = 'product-card';
                            
                            let itemId = typeof item === 'object' ? (item.product_id || item.item_id || item.id) : item;
                            let score = typeof item === 'object' ? item.score : null;

                            let html = `<div class="product-id">📦 Item ID: ${itemId}</div>`;
                            html += `<div class="product-score"><span>Hạng: #${index + 1}</span>`;
                            if (score !== null && score !== undefined) {
                                let scoreLabel = data.fallback ? 'Tương tác' : 'Điểm Ranking';
                                let formattedScore = data.fallback ? parseInt(score) : parseFloat(score).toFixed(4);
                                html += `<span>${scoreLabel}: ${formattedScore}</span>`;
                            }
                            html += `</div>`;
                            
                            card.innerHTML = html;
                            productsGrid.appendChild(card);
                        });
                    } else {
                        productsGrid.innerHTML = '<p>Không tìm thấy thông tin sản phẩm cụ thể. Bạn có thể tham khảo Raw Data bên dưới.</p>';
                        rawDataContainer.style.display = 'block';
                        rawJson.textContent = JSON.stringify(data, null, 2);
                    }

                } catch (err) {
                    errorBox.textContent = '❌ Lỗi: ' + err.message;
                    errorBox.style.display = 'block';
                } finally {
                    btnText.style.display = 'block';
                    loader.style.display = 'none';
                    btn.disabled = false;
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/recommend/{user_id}")
def get_recommendations(user_id: int, top_n: int = 20, session_items: Optional[str] = None):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline is not initialized yet.")
        
    try:
        parsed_session_items = None
        if session_items:
            try:
                parsed_session_items = [int(x.strip()) for x in session_items.split(",") if x.strip()]
            except ValueError:
                raise HTTPException(status_code=400, detail="session_items must be a comma-separated list of integers.")
                
        result = pipeline.recommend(user_id=user_id, top_n=top_n, session_items=parsed_session_items)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend-by-index/{user_idx}")
def get_recommendations_by_index(user_idx: int, top_n: int = 20):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline is not initialized yet.")

    try:
        result = pipeline.recommend_by_index(user_idx=user_idx, top_n=top_n)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main_serve:app", host="127.0.0.1", port=8000, reload=False)

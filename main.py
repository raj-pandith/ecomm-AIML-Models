from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import os
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
from decimal import Decimal
import uvicorn

from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="AI Pricing & Recs Demo")

# ====================
# CONFIG & GLOBALS
# ====================
MODELS_DIR = "models"
PRICING_MODEL_PATH = os.path.join(MODELS_DIR, "pricing_model.joblib")
RECOMMENDER_MODEL_PATH = os.path.join(MODELS_DIR, "recommender_model.joblib")

pricing_info = None
recommender = None
embedder = None
PRODUCT_EMBEDDINGS = {}

# ====================
# DATABASE CONNECTION
# ====================
def get_db_engine():
    database_url = os.getenv("DATABASE_URL")
    return create_engine(database_url, pool_pre_ping=True)


@app.get("/")
def home():
    return {"message": "Hello from FastAPI !"}

# ====================
# STARTUP EVENT
# ====================
# @app.on_event("startup")
# async def startup_event():
#     global pricing_info, recommender, embedder, PRODUCT_EMBEDDINGS

#     print("\n=== Application Startup Started ===")

#     # ---- Pricing model
#     if os.path.exists(PRICING_MODEL_PATH):
#         pricing_info = joblib.load(PRICING_MODEL_PATH)
#         print("✅ Pricing model loaded")
#     else:
#         print("⚠️ Pricing model not found")

#     # ---- Recommender model
#     if os.path.exists(RECOMMENDER_MODEL_PATH):
#         recommender = joblib.load(RECOMMENDER_MODEL_PATH)
#         print("✅ Recommender model loaded")
#     else:
#         print("⚠️ Recommender model not found")

#     # ---- Embedding model
#     try:
#         embedder = SentenceTransformer("all-MiniLM-L6-v2")
#         print("✅ SentenceTransformer loaded")
#     except Exception as e:
#         print(f"❌ Embedding model load failed: {e}")
#         embedder = None

#     # ---- Build embeddings
#     PRODUCT_EMBEDDINGS = {}

#     if embedder is None:
#         print("⚠️ Skipping embeddings (no embedder)")
#         return

#     try:
#         engine = get_db_engine()
#         print("Connecting to database...")

#         with engine.connect() as conn:
#             rows = conn.execute(
#                 text("SELECT id, name, description FROM products")
#             ).fetchall()

#         print(f"Found {len(rows)} products")

#         for idx, row in enumerate(rows, 1):
#             pid = None
#             product_text = ""

#             try:
#                 pid = int(row[0])
#                 name = row[1] or ""
#                 desc = row[2] or ""

#                 product_text = f"{name} {desc}".strip()
#                 if not product_text:
#                     continue

#                 emb = embedder.encode(product_text)
#                 PRODUCT_EMBEDDINGS[pid] = emb.tolist()

#                 print(f"  ✔ Embedded product {pid}")

#             except Exception as row_err:
#                 print(
#                     f"  ❌ Row {idx} failed | id={pid} | "
#                     f"text='{product_text[:40]}' | err={row_err}"
#                 )

#         print(f"✅ Created embeddings for {len(PRODUCT_EMBEDDINGS)} products")

#     except Exception as e:
#         print(f"❌ Embedding build failed: {e}")
#         PRODUCT_EMBEDDINGS = {}

#     print("=== Application Startup Completed ===\n")

pricing_info = None
recommender = None

def get_pricing_model():
    global pricing_info
    if pricing_info is None and os.path.exists(PRICING_MODEL_PATH):
        pricing_info = joblib.load(PRICING_MODEL_PATH)
        print("✅ Pricing model loaded (lazy)")
    return pricing_info


def get_recommender():
    global recommender
    if recommender is None and os.path.exists(RECOMMENDER_MODEL_PATH):
        recommender = joblib.load(RECOMMENDER_MODEL_PATH)
        print("✅ Recommender model loaded (lazy)")
    return recommender
# ====================
# USER-BASED RECOMMENDATION
# ====================
@app.get("/recommend")
def recommend(user_id: int, n: int = 6):
    if recommender is None:
        raise HTTPException(500, "Recommender model not loaded")

    preds = [
        (pid, recommender.predict(user_id, pid).est)
        for pid in range(1, 26)
    ]

    top_n = sorted(preds, key=lambda x: x[1], reverse=True)[:n]

    return {
        "user_id": user_id,
        "recommended_product_ids": [pid for pid, _ in top_n]
    }

# ====================
# PERSONALIZED PRICING
# ====================

@app.get("/price")
def get_price(user_id: int, product_id: int):
    engine = get_db_engine()

    try:
        with engine.connect() as conn:
            user_row = conn.execute(
                text("SELECT loyalty_points FROM users WHERE id=:uid"),
                {"uid": user_id}
            ).fetchone()

            if not user_row:
                raise HTTPException(404, "User not found")

            loyalty_points = int(user_row[0])

            product_row = conn.execute(
                text("""
                    SELECT base_price, sales_count, category
                    FROM products WHERE id=:pid
                """),
                {"pid": product_id}
            ).fetchone()

            if not product_row:
                raise HTTPException(404, "Product not found")

        base_price_raw, sales_count_raw, category_raw = product_row

        base_price = float(base_price_raw) if base_price_raw is not None else 0.0
        sales_count = int(sales_count_raw) if sales_count_raw is not None else 0
        category = str(category_raw) if category_raw is not None else "Unknown"

        if pricing_info is None:
            return {
                "suggested_price": round(base_price, 2),
                "discount_percent": 0.0,
                "reason": "Pricing model not loaded - showing full price"
            }

        model = pricing_info["model"]
        encoder = pricing_info["encoder"]

        cat_encoded = (
            encoder.transform([category])[0]
            if category in encoder.classes_
            else 0
        )

        X = np.array([[loyalty_points, sales_count, cat_encoded]], dtype=float)

        discount_frac = model.predict(X)[0]
        discount_percent = min(max(float(discount_frac) * 100, 0), 35)

        # Force minimum discount based on loyalty (guaranteed non-zero for demo)
        min_discount = loyalty_points / 100.0  # e.g. 200 → 2.0%
        discount_percent = max(discount_percent, min_discount)

        suggested_price = base_price * (1 - discount_percent / 100)

        # Full debug prints
        print(f"DEBUG - User {user_id}: loyalty_points = {loyalty_points}")
        print(f"DEBUG - Product {product_id}: base_price = {base_price}, sales_count = {sales_count}, category = '{category}'")
        print(f"DEBUG - Category encoded = {cat_encoded}")
        print(f"DEBUG - Features sent to model: {X.tolist()}")
        print(f"DEBUG - Raw discount_frac from model: {discount_frac:.6f}")
        print(f"DEBUG - Discount after model clipping: {discount_percent:.2f}%")
        print(f"DEBUG - Forced min discount from loyalty: {min_discount:.2f}%")
        print(f"DEBUG - Final discount_percent used: {discount_percent:.2f}%")
        print(f"DEBUG - Calculated suggested_price: {suggested_price:.2f}")

        return {
            "suggested_price": round(suggested_price, 2),
            "discount_percent": round(discount_percent, 1),
            "reason": f"Based on loyalty {loyalty_points} pts + demand {sales_count} sales"
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /price endpoint: {e}")
        raise HTTPException(500, str(e))

# ====================
# CONTENT-BASED SIMILAR PRODUCTS
# ====================
@app.get("/recommend-similar")
def recommend_similar(product_id: int, n: int = 6):
    if product_id not in PRODUCT_EMBEDDINGS:
        raise HTTPException(404, "Product not found or embeddings missing")

    query = np.array(PRODUCT_EMBEDDINGS[product_id])
    sims = []

    for pid, emb in PRODUCT_EMBEDDINGS.items():
        if pid == product_id:
            continue
        emb = np.array(emb)
        sim = np.dot(query, emb) / (
            np.linalg.norm(query) * np.linalg.norm(emb) + 1e-8
        )
        sims.append((pid, sim))

    top_n = sorted(sims, key=lambda x: x[1], reverse=True)[:n]

    return {
        "based_on_product": product_id,
        "recommended_product_ids": [pid for pid, _ in top_n]
    }




@app.get("/search")
def search_products(query: str, n: int = 6):
    if not query or not query.strip():
        return {"error": "Query cannot be empty"}

    if PRODUCT_EMBEDDINGS is None or len(PRODUCT_EMBEDDINGS) == 0:
        return {"error": "No product embeddings loaded. Run /embed-products first or check startup logs."}

    try:
        # Encode the search query
        query_text = query.strip()
        query_embedding = embedder.encode(query_text)

        # Calculate similarity with all products
        similarities = []
        for pid, emb in PRODUCT_EMBEDDINGS.items():
            emb_array = np.array(emb)
            sim = np.dot(query_embedding, emb_array) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(emb_array) + 1e-8
            )
            similarities.append((pid, float(sim)))

        # Sort by similarity score (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Take top N
        top_n = similarities[:n]

        return {
            "query": query_text,
            "results": [
                {
                    "product_id": pid,
                    "similarity_score": round(score, 4)
                }
                for pid, score in top_n
            ],
            "total_results": len(top_n),
            "message": f"Top {len(top_n)} semantically similar products for '{query_text}'"
        }

    except Exception as e:
        print(f"Search error: {e}")
        return {"error": str(e)}






if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
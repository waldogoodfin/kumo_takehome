# ShopSight – Agentic E-commerce Analytics Prototype

## Overview
ShopSight transforms how merchandising teams make product decisions by turning natural language queries into actionable business intelligence. Instead of spending hours digging through spreadsheets and dashboards, merchandisers can simply type "Lisbon tencel dress" and instantly get:

- **Smart Product Discovery**: AI-powered search that understands intent, not just keywords
- **Complete Performance Picture**: Sales trends, forecasts, and customer segments in one view
- **What-If Planning**: Counterfactual scenarios (Sandbox module) to test marketing strategies before launch
- **Strategic Recommendations**: AI generated Consultant-level insights with specific tactics and budget allocations
- **Ready-to-Use Insights**: Professional summaries that can be dropped directly into planning documents

**The Problem**: Merchandising teams waste countless hours switching between tools, manually analyzing data, and translating numbers into actionable strategies.

**Our Solution**: One search query delivers everything needed to make confident product decisions—from performance data to strategic recommendations.

---

## My Main Focus -> Product Strategy & User-Centered Design

### **1. Solve the Core User Pain Point**
**User Need**: Merchandisers need to quickly understand "Should I invest more in this product?" and "How should I market it?"

**Our Approach**: 
- Built a seamless "search → comprehensive insights" flow that answers both questions in under 10 seconds
- Combined lexical and semantic search (using OpenAI embeddings) so users can search naturally ("winter coat" or "sustainable dress under $500")
- Prioritized the insight modal as the hero experience—everything a merchandiser needs in one focused view

### **2. Make Data Feel Real and Actionable**
**User Need**: Generic analytics feel disconnected from real business decisions

**Our Approach**:
- Used authentic H&M catalog data (105k products) so every search feels realistic
- Generated structured transaction data that creates believable sales trends and forecasts
- Focused on specific, actionable recommendations ("Allocate $15K to Instagram ads targeting 25-34 demographics") rather than vague insights

### **3. Eliminate Cognitive Load**
**User Need**: Merchandisers are overwhelmed by switching between multiple tools and interpreting raw data

**Our Approach**:
- Single search interface that replaces multiple dashboards
- AI-generated insights that translate numbers into business language
- Clean, focused UI that presents complex data without overwhelming the user
- Counterfactual scenarios that let users explore "what-if" questions without building complex models

### **4. Build for Scale and Evolution**
**User Need**: Tools that grow with business needs and don't become obsolete

**Our Approach**:
- Modular FastAPI architecture allows easy integration of real forecasting models, vector databases, or additional data sources
- LLM-powered insights can adapt to new business contexts and product categories
- Designed API structure to support future agentic workflows where AI orchestrates multiple business tools

---

## Architecture & Flow
```
┌──────────────┐      ┌────────────────────────────┐      ┌─────────────────┐
│  React Front │◄────►│ FastAPI (modular services) │◄────►│ H&M Catalog +   │
│  - Hero & UX │      │  - search.py               │      │ Customers (real)│
│  - Insights   │     │  - analytics.py            │      │ + Mock Sales    │
│  - Dashboard  │     │  - embeddings.py           │      └─────────────────┘
└──────────────┘      │  - llm.py (GPT-4.1-mini)   │
                      └────────────────────────────┘
                              ▲
                              │
                    OpenAI API (chat + embeddings)
```
**Pipeline:**
1. **Search intent** → LLM normalizes the query & suggests filters.
2. **Hybrid ranking** → RapidFuzz lexical + OpenAI embeddings.
3. **Insights service** → Sales trend, forecast, segments, counterfactuals.
4. **Narrative** → GPT-4.1-mini turns numbers into bullet-point actions.
5. **UI** → Hero panel, search grid, and insights modal stitch it together.

---

## What’s Real vs. Mocked
| Area | Status | Notes |
|------|--------|-------|
| Product catalog | ✅ Real | 105k SKUs + metadata from H&M parquet
| Customer base | ✅ Real | 1.37M hashed IDs & attributes
| Natural-language search | ✅ Real | LLM intent + hybrid ranking pipeline
| Sales history chart | ✅ Real-ish | Derived from synthetic but structured transactions
| Forecast values | 🎭 Mocked | Statistical stub (avg revenue ± shake) with confidence labels, but uses real transaction density
| Customer segments | 🎭 Mocked | Fixed personas (Fashion Enthusiasts 42%, Value Seekers 33%, Trend Followers 25%) sized to look believable
| Counterfactual scenarios | ✅ Real-ish | Dynamic baseline calculation using actual forecast + predefined campaign levers with consistent uplift math
| "Customers also bought" | 🎭 Mocked | Sampled by department; ready for real co-buy matrix later
| Agent narration | ✅ Real | GPT-4.1-mini generates aggressive, consultant-level insights with trend analysis and specific tactics
| Similar product retrieval | 🎭 Hybrid | Department-based sampling; plug in embeddings later

---

## Assumptions
- Marketing/merch teams want headline answers, not just raw charts.
- Six months of synthetic transactions is enough to illustrate trends without overwhelming the mock generator.
- GPT-4.1-mini is available (if not, swap `SHOPSIGHT_CHAT_MODEL`).
- For demo speed, we embed only a sample (~7.5k items) unless the vector toggle is off.

---

## Running the Demo Locally
### Prerequisites
- Python 3.11+ (tested on 3.12)
- Node 18+
- OpenAI API key

### 1. Clone & Install
```bash
cd shopsight
pip install -r requirements.txt
npm --prefix frontend install
```

### 2. Configure Environment
Create a `.env` file in the `shopsight` directory:
```bash
# Required for LLM features
OPENAI_API_KEY=sk-your-key-here

# Optional configuration
SHOPSIGHT_ENABLE_EMBEDDINGS=true     # Enable semantic search (requires OpenAI key)
SHOPSIGHT_CHAT_MODEL=gpt-4.1-mini    # LLM model for insights generation
```

### 3. Start Backend
```bash
cd backend
python main.py  # or uvicorn main:app --reload
```

### 4. Start Frontend
```bash
cd ../frontend
npm start
```

### 5. Browse
- App: http://localhost:3000
- API docs: http://localhost:8000/docs

---

## To Try the System

### 🎯 **Hero Demo Flow**
1. **Open the hero page** → http://localhost:3000
2. **Search "Lisbon tencel dress"** → Shows hybrid search with LLM intent parsing
3. **Click the top result** → Opens comprehensive insights modal with:
   - **Sales Trend**: "Sales increased by $242.29 last month" 
   - **Next Month Forecast**: $512.15 (Medium confidence)
   - **Customer Segments**: Fashion Enthusiasts (42%), Value Seekers (33%), Trend Followers (25%)
   - **AI Insights**: Aggressive, consultant-level recommendations with specific budgets and tactics
   - **Counterfactual Scenarios**: Influencer Boost (+18%), Bundle Offer (+12%), Geo-Targeted Ads (+8%)
   - **Similar Products**: Department and type-based recommendations

### 🔄 **Alternative Demo Queries**
- **"RIHANNA dress"** → Shows graceful handling of products with no sales data
- **"winter coat under 500"** → Demonstrates LLM intent classification and price filtering
- **"dress"** → Shows multiple product options and search variety

### 📊 **Dashboard Overview**
- **Jump to dashboard tab** → Aggregate KPIs, top performers chart, business metrics

---

## 🧠 How the LLM is Used

### **Core LLM Functions**
1. **Intent parsing** – Converts open-text queries (e.g., "Lisbon tencel dress under $500") into structured search hints with normalized queries, filters, and preferred departments.
2. **Aggressive insight narration** – Transforms raw metrics into consultant-level, actionable bullets with specific budgets, tactics, and growth targets.
3. **Trend-aware analysis** – References momentum direction (growing/declining/stable) and incorporates actual performance numbers into recommendations.
4. **Clean text output** – Returns plain text insights without markdown formatting for seamless UI display.

### **Enhanced Capabilities**
- **Data-driven specificity**: Includes actual numbers ($242.29 increase, $512.15 forecast, 25% inventory targets)
- **Channel-specific tactics**: Recommends precise platforms (Instagram, TikTok, Google Shopping) with audience targeting
- **Performance-based baselines**: Generates appropriate counterfactual baselines even for products with no sales data

### **Agentic Architecture**
The insights service acts as a modular toolchain: **sales trend → forecast → segments → scenarios → narrative**. Each component is independently callable, making it straightforward to add a coordinator LLM layer that orchestrates these tools based on user intent and data availability.

---

## Gaps & How We'd Fill Them

### **Data & Intelligence Layer**

**Real Transaction Ingestion**
Currently using synthetic transaction data that's structured but not real. Going forward, we'd need to hook into actual order feeds or data warehouses. The good news is our data loader in `data/loader.py` is designed to be swapped out easily - we'd replace the mock generator with ETL pipelines that can handle real-time or batch data ingestion from systems like Shopify, BigCommerce, or internal databases.

**Forecast Accuracy** 
Our current forecasting is a statistical stub that averages historical performance with some randomness. For production, we'd implement proper time series forecasting using Prophet, LSTM networks, or ideally **Kumo's ML platform** which would be perfect for this use case. The interface is already designed to return horizon arrays, so swapping in real models wouldn't break the UI.

**Customer Segment Intelligence**
Right now we show fixed personas (Fashion Enthusiasts 42%, Value Seekers 33%, etc.) that look realistic but don't adapt. Real segmentation would involve clustering analysis on RFM metrics, purchase behavior, and demographic data. We'd want to understand not just who buys what, but why they buy it and how to reach similar customers more effectively.

### **Search & Discovery**

**Semantic Search at Scale**
We're currently using OpenAI embeddings for a sample of products, but for millions of SKUs we'd need a proper vector database like Pinecone or Weaviate. The hybrid search architecture is already there - we'd just need to scale the embedding generation and storage to handle full catalogs with sub-second response times.

**Similar Product & Co-bought Logic**
Our "similar products" are currently sampled by department, which works for demo but isn't very intelligent. Real similarity would use product embeddings, collaborative filtering, and association rule mining to find products that actually complement each other or serve as substitutes. This would enable proper cross-selling and bundle recommendations.

### **User Experience & Trust**

**Agent UI Transparency**
Users need to understand how the AI arrived at its recommendations. We'd add an "Agent timeline" or "reasoning path" that shows which data sources were consulted, what analysis was performed, and why specific insights were generated. This builds trust and helps users debug when recommendations seem off.

**Interactive Counterfactual Scenarios**
The counterfactual sandbox currently shows static scenarios, but users want to play with the assumptions. We'd wire the `/product/{id}/counterfactuals` endpoint into interactive charts where users can adjust campaign budgets, target audiences, or seasonal factors and see the projected impact update in real-time.

### **Production & Scale**

**LLM Cost & Performance Optimization**
OpenAI API calls add up quickly. We'd implement intelligent caching for frequent insights, batch API calls where possible, and potentially fine-tune smaller models for specific tasks. Response streaming would make the UI feel faster even when LLM calls take time.

**Security & Enterprise Features**
For enterprise deployment, we'd need OAuth integration, role-based access control, audit logging, and data governance features. The FastAPI structure already supports middleware, so adding these wouldn't require architectural changes.

### **Strategic Evolution**

**Full Agentic Workflows**
The current system acts like a smart toolchain - search leads to analytics leads to insights. The next evolution would be a coordinator LLM that can orchestrate multiple tools based on user intent. Imagine asking "How should I position our winter collection?" and having the system automatically analyze seasonal trends, competitor pricing, inventory levels, and customer segments to generate a comprehensive strategy.

**Multi-tenant & White-label**
The modular architecture could easily support multiple retailers with different data sources, branding, and business rules. Each tenant could have their own models, segments, and insight templates while sharing the core infrastructure.


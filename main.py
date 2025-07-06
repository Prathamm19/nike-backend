import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import requests  # For Together.ai API
from fastapi import FastAPI



load_dotenv()

# Initialize components
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
pc = pinecone.Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index("nike")

# Together.ai config
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')
TOGETHER_ENDPOINT = "https://api.together.xyz/v1/completions"

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or list specific domain: ["https://nike-search-git-main-prathamms-projects.vercel.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class LLMRequest(BaseModel):
    product_id: str
    user_question: str = None

def get_product_metadata(product_id: str):
    """Fetch stored product data from Pinecone"""
    return index.fetch(ids=[product_id]).get('vectors', {}).get(product_id, {}).get('metadata', {})



from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse



@app.api_route("/", methods=["GET", "HEAD"])
async def root(request: Request):
    # Optional: don't return full body on HEAD request
    if request.method == "HEAD":
        return JSONResponse(status_code=200)
    return {"message": "Backend is running 🚀"}


@app.post("/search")
async def search(request: SearchRequest):
    try:
        query_embed = embed_model.encode(request.query).tolist()
        results = index.query(
            vector=query_embed,
            top_k=request.top_k,
            include_metadata=True
        )
        return {
            "results": [
                {
                    "id": match["id"],
                    "title": match["metadata"]["title"],
                    "score": match["score"]
                }
                for match in results["matches"]
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#@app.post("/ask_llm")
#async def ask_llm(request: LLMRequest):
#    """Augment product info with Together.ai"""
#    try:
#        product = get_product_metadata(request.product_id)
 #          raise HTTPException(status_code=404, detail="Product not found")
#
#        prompt = f"""
#        Product: {product['title']} 
#        Description: {product['description']}
#        
#        Question: {request.user_question or 'Summarize key features'}
 #       Answer:
  #      """
   #     
    #       TOGETHER_ENDPOINT,
#            headers={"Authorization": f"Bearer {TOGETHER_API_KEY}"},
 #           json={
  #              "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
   #             "prompt": prompt,
    #            "max_tokens": 200
     #       }
      #  )
       # return {
        #    "product": product,
         #   "llm_response": response.json()["choices"][0]["text"]
        #}
    #except Exception as e:
     #   raise HTTPException(status_code=500, detail=str(e))

@app.post("/product")
async def get_product(request: LLMRequest):
    try:
        print("🛬 Called /product with:", request.product_id)

        product_data = index.fetch([request.product_id])
        print("📦 Fetched:", product_data)

        if not product_data.vectors:
            raise HTTPException(status_code=404, detail="Product not found")

        vector = product_data.vectors.get(request.product_id)
        if not vector:
            raise HTTPException(status_code=404, detail="Vector not found")

        # ✅ Access metadata using dot-notation
        metadata = vector.metadata or {}

        title = metadata.get("title", "Unknown Product")
        description = metadata.get("description", "No description available")
        subtitle = metadata.get("subtitle", "")

        print("✅ Returning product:", title)
        return {
    "title": title,
    "subtitle": subtitle,     # ✅ Include it in response
    "description": description
}

    except Exception as e:
        print("🔥 Error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
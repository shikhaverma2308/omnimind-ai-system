import streamlit as st
from pypdf import PdfReader
import faiss
import numpy as np
import ollama
import time
from PIL import Image

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="God Mode AI System", layout="centered")
st.title("🔥 OmniMind: Integrated Agentic Multimodal System")

uploaded_file = st.file_uploader("📄 Upload PDF", type="pdf")
uploaded_image = st.file_uploader("🖼 Upload Image", type=["png", "jpg", "jpeg"])

query = st.text_input("💬 Ask questions?")

mode = st.selectbox("⚡ Mode", ["Fast", "Accurate"])
user_type = st.selectbox("👤 User Type", ["Beginner", "Advanced"])

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------
# MODEL SELECT
# ---------------------------
model_name = "phi" if mode == "Fast" else "llama3"

# ---------------------------
# PDF LOAD
# ---------------------------
def load_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text

# ---------------------------
# CHUNKING
# ---------------------------
def chunk_text(text, chunk_size=1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# ---------------------------
# EMBEDDINGS
# ---------------------------
def get_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        res = ollama.embeddings(
            model="nomic-embed-text",
            prompt=chunk
        )
        embeddings.append(res["embedding"])
    return np.array(embeddings).astype("float32")

# ---------------------------
# VECTOR DB
# ---------------------------
@st.cache_resource
def build_index(chunks_tuple):
    chunks = list(chunks_tuple)
    embeddings = get_embeddings(chunks)
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# ---------------------------
# ROUTER
# ---------------------------
def router(query):
    if uploaded_image:
        return "image"
    elif uploaded_file:
        return "rag"
    else:
        return "general"

# ---------------------------
# INTENT DETECTOR
# ---------------------------
def intent_detector(query):
    q = query.lower()
    if "summary" in q:
        return "summary"
    elif "detail" in q or "explain" in q:
        return "detailed"
    else:
        return "normal"

# ---------------------------
# RETRIEVER
# ---------------------------
def retriever(query, index, chunks):
    q_embed = ollama.embeddings(
        model="nomic-embed-text",
        prompt=query
    )["embedding"]

    q_embed = np.array([q_embed]).astype("float32")
    distances, I = index.search(q_embed, k=3)

    retrieved = [chunks[i] for i in I[0]]
    return "\n".join(retrieved), I[0], distances[0]

# ---------------------------
# EXPLAINER
# ---------------------------
def explain(context, query):
    style = "Explain in simple words" if user_type == "Beginner" else "Give technical explanation"

    prompt = f"""
{style}

Context:
{context}

Question:
{query}

Answer step-by-step.
"""

    try:
        res = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return res["message"]["content"]
    except:
        return "⚠ Low memory issue"

## SUMMARIZER
# ---------------------------
def summarize(answer):
    res = ollama.chat(
        model=model_name,
        messages=[{
            "role": "user",
            "content": f"Summarize in 2 lines:\n{answer}"
        }]
    )
    return res["message"]["content"]

# ---------------------------
# IMAGE AGENT
# ---------------------------
def analyze_image(image_file):
    try:
        res = ollama.chat(
            model="llava",
            messages=[{
                "role": "user",
                "content": "Explain this image",
                "images": [image_file.getvalue()]
            }]
        )
        return res["message"]["content"]
    except:
        return "⚠ Not enough RAM"

# ---------------------------
# SELF REFLECTION
# ---------------------------
def self_reflect(answer):
    res = ollama.chat(
        model=model_name,
        messages=[{
            "role": "user",
            "content": f"Improve this answer if needed:\n{answer}"
        }]
    )
    return res["message"]["content"]

# ---------------------------
# MAIN SYSTEM
# ---------------------------
if st.button("🚀 Ask") and query:

    start = time.time()

    route = router(query)
    intent = intent_detector(query)

    with st.spinner("🤖 AI thinking..."):

        # IMAGE FLOW
        if route == "image":
            st.image(uploaded_image, caption="Uploaded Image")
            answer = analyze_image(uploaded_image)

        # RAG FLOW
        elif route == "rag":

            text = load_pdf(uploaded_file)

            if not text.strip():
                st.error("❌ No text found")
                st.stop()

            chunks = chunk_text(text)
            index = build_index(tuple(chunks))

            context, indices, distances = retriever(query, index, chunks)

            # FALLBACK
            if len(context.strip()) < 50:
                st.warning("⚠ Not found in PDF → switching to general AI")
                res = ollama.chat(
                    model=model_name,
                    messages=[{"role": "user", "content": query}]
                )
                answer = res["message"]["content"]
            else:
                detailed = explain(context, query)

                if intent == "summary":
                    answer = summarize(detailed)
                elif intent == "detailed":
                    answer = detailed
                else:
                    answer = summarize(detailed)

                # SELF REFLECTION
                answer = self_reflect(answer)

                # SOURCES
                st.subheader("📚 Sources")
                for i, idx in enumerate(indices):
                    st.write(f"Source {i+1}: {chunks[idx][:200]}")

                # CONFIDENCE
                confidence = round(1 / (distances[0] + 1), 2)
                st.write(f"🔎 Confidence Score: {confidence}")

        # GENERAL FLOW
        else:
            res = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": query}]
            )
            answer = res["message"]["content"]

    end = time.time()

    # STREAMING OUTPUT
    st.subheader("💡 Answer")
    placeholder = st.empty()
    text_stream = ""

    for word in answer.split():
        text_stream += word + " "
        placeholder.write(text_stream)

    # PERFORMANCE
    st.write(f"⏱ Response Time: {round(end - start, 2)} sec")

    # MEMORY
    st.session_state.history.append({"q": query, "a": answer})

# ---------------------------
# MEMORY UI
# ---------------------------
st.subheader("🧠 Memory")

for chat in st.session_state.history[::-1]:
    st.write(f"Q: {chat['q']}")
    st.write(f"A: {chat['a']}")
    st.write("---")

# ---------------------------
# ARCHITECTURE
# ---------------------------
st.markdown("## ⚙ System Flow")
st.markdown("""
User → Router → (RAG / Image / General) → Intent → Agents → Reflection → Answer
""")
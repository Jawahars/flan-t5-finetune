import os
import zipfile
import pickle
import faiss
import torch
import numpy as np
import warnings
import time
import pandas as pd
import gradio as gr
import shutil

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from peft import PeftModel

# --- 1. SETUP AND CONFIGURATION ---

warnings.filterwarnings("ignore", message="Already found a `peft_config` attribute")

def setup_data_directory(zip_path, data_path):
    """
    Checks for the data directory. If it's missing, this function
    CREATES the directory and then EXTRACTS the zip file's
    contents directly into it.
    """
    if os.path.exists(data_path):
        print(f"✅ '{data_path}' directory already exists. Skipping setup.")
        return True

    if not os.path.exists(zip_path):
        print(f"FATAL ERROR: '{zip_path}' not found. Please place it in the same folder as app.py.")
        return False

    try:
        # 1. Create the target 'data' directory first.
        print(f"'{data_path}' not found. Creating it...")
        os.makedirs(data_path, exist_ok=True)

        # 2. Extract the zip contents directly into the new 'data' directory.
        print(f"Extracting '{zip_path}' into '{data_path}'...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        
        # 3. Verify that the extraction was successful.
        expected_subfolder = os.path.join(data_path, 'rag_model_artifacts')
        if os.path.exists(expected_subfolder):
            print(f"✅ Successfully extracted zip contents into '{data_path}'.")
            return True
        else:
            print(f"FATAL ERROR: Extraction failed to create expected subfolder '{expected_subfolder}'.")
            shutil.rmtree(data_path) # Clean up empty directory
            return False

    except Exception as e:
        print(f"FATAL ERROR: An error occurred during extraction: {e}")
        if os.path.exists(data_path):
            shutil.rmtree(data_path)
        return False

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"🚀 Using device: {DEVICE}")

# --- 2. LOAD ALL MODELS AND ARTIFACTS (Defined as Functions) ---

def load_rag_system(data_path):
    print("\n--- Loading RAG System Components ---")
    RAG_ARTIFACTS_PATH = os.path.join(data_path, "rag_model_artifacts")
    
    faiss_idx = faiss.read_index(os.path.join(RAG_ARTIFACTS_PATH, "faiss.index"))
    print("✅ FAISS index loaded.")

    with open(os.path.join(RAG_ARTIFACTS_PATH, "bm25.pkl"), "rb") as f:
        bm25_idx = pickle.load(f)
    print("✅ BM25 index loaded.")

    with open(os.path.join(RAG_ARTIFACTS_PATH, "all_chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)
    print(f"✅ Loaded {len(chunks)} text chunks.")

    emb_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=DEVICE)
    print("✅ Embedding model loaded.")

    cross_enc = CrossEncoder(os.path.join(RAG_ARTIFACTS_PATH, 'cross_encoder'), device=DEVICE)
    print("✅ Cross-Encoder model loaded.")

    gen_tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
    gen_mod = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(DEVICE)
    print("✅ Generator model for RAG loaded.")
    
    return faiss_idx, bm25_idx, chunks, emb_model, cross_enc, gen_tok, gen_mod

def load_finetuned_system(data_path):
    print("\n--- Loading Fine-Tuned Model Components ---")
    FINETUNED_ARTIFACTS_PATH = os.path.join(data_path, "finetuned_model_artifacts")

    base_model_ft = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(DEVICE)
    moe_tok = AutoTokenizer.from_pretrained(os.path.join(FINETUNED_ARTIFACTS_PATH, "tokenizer"))
    print("✅ Base model for MoE and tokenizer loaded.")

    expert_bs = PeftModel.from_pretrained(base_model_ft, os.path.join(FINETUNED_ARTIFACTS_PATH, "lora_expert_balance_sheet"))
    print("✅ Balance Sheet expert (LoRA adapter) loaded.")

    expert_is = PeftModel.from_pretrained(base_model_ft, os.path.join(FINETUNED_ARTIFACTS_PATH, "lora_expert_income_statement"))
    print("✅ Income Statement expert (LoRA adapter) loaded.")

    return moe_tok, expert_bs, expert_is

# --- 3. DEFINE INFERENCE AND UI BACKEND FUNCTIONS ---

def query_finance_system_rag(query: str, top_n=5):
    query_embedding = embedding_model.encode([query.lower()])
    _, dense_indices = faiss_index.search(np.array(query_embedding).astype('float32'), top_n)
    tokenized_query = query.lower().split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    sparse_indices = np.argsort(bm25_scores)[::-1][:top_n]
    combined_indices = sorted(list(set(dense_indices[0]) | set(sparse_indices)))
    retrieved_chunks = [all_chunks[i] for i in combined_indices]
    cross_input = [[query, chunk['text']] for chunk in retrieved_chunks]
    cross_scores = cross_encoder.predict(cross_input, show_progress_bar=False)
    for i, chunk in enumerate(retrieved_chunks): chunk['relevance_score'] = cross_scores[i]
    reranked_chunks = sorted(retrieved_chunks, key=lambda x: x['relevance_score'], reverse=True)
    context = "\n\n".join([chunk.get("text", "") for chunk in reranked_chunks[:3]])
    prompt = f"Based on the context, answer the question.\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
    inputs = gen_tokenizer(prompt, return_tensors="pt").to(DEVICE)
    output = gen_model.generate(**inputs, max_new_tokens=100)
    return gen_tokenizer.decode(output[0], skip_special_tokens=True)

def query_finance_system_finetune(question: str):
    q_lower = question.lower()
    model_to_use = expert_balance_sheet if 'assets' in q_lower or 'liabilities' in q_lower else expert_income_statement
    inputs = moe_tokenizer(question, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model_to_use.generate(**inputs, max_new_tokens=50)
    return moe_tokenizer.decode(outputs[0], skip_special_tokens=True)

def _run_backend(question: str, mode: str) -> str:
    q = (question or "").strip()
    if not q: return "Please enter a question."
    print(f"Received query: '{q}' for mode: {mode}")
    return query_finance_system_rag(q) if mode == "RAG" else query_finance_system_finetune(q)

def answer_and_log(question: str, mode: str, history: list):
    answer_text = _run_backend(question, mode)
    entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "method": mode, "question": (question or "").strip(), "answer": answer_text}
    history = (history or []) + [entry]
    chatbot_pairs = [(h["question"], f"[{h['method']}] {h['answer']}") for h in history]
    hist_df = pd.DataFrame(history, columns=["time", "method", "question", "answer"])
    return f"[{mode}] {answer_text}", history, chatbot_pairs, hist_df

def clear_history():
    return "", [], [], pd.DataFrame(columns=["time", "method", "question", "answer"])

# --- 4. LAUNCH THE GRADIO USER INTERFACE ---
def create_ui():
    with gr.Blocks(title="Financial QA", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## Financial QA - RAG vs Fine-Tuned\nAsk a question about the Kyndryl financial reports and see history below.")
        state_history = gr.State([])
        with gr.Row():
            with gr.Column(scale=5):
                question_in = gr.Textbox(label="Question", placeholder="e.g., What were the total revenues in fiscal 2024?", lines=2)
                mode_in = gr.Radio(choices=["RAG", "Fine-Tune"], value="RAG", label="Method")
                with gr.Row():
                    ask_btn = gr.Button("Answer", variant="primary")
                    clear_btn = gr.Button("Clear History")
            with gr.Column(scale=5):
                answer_out = gr.Markdown(label="Answer")
                chat_out = gr.Chatbot(label="Q/A History (Chat view)", bubble_full_width=False)
        table_out = gr.Dataframe(label="History (Table view)", interactive=False)
        ask_btn.click(fn=answer_and_log, inputs=[question_in, mode_in, state_history], outputs=[answer_out, state_history, chat_out, table_out])
        clear_btn.click(fn=clear_history, inputs=None, outputs=[answer_out, state_history, chat_out, table_out])
    return demo

# --- 5. MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # Define constants for file paths
    DATA_DIR = "data"
    ZIP_FILE = "data_backup.zip"
    
    # First, set up the data directory by passing the constants to the function
    if setup_data_directory(zip_path=ZIP_FILE, data_path=DATA_DIR):
        # If setup is successful, load all models into global variables
        faiss_index, bm25, all_chunks, embedding_model, cross_encoder, gen_tokenizer, gen_model = load_rag_system(data_path=DATA_DIR)
        moe_tokenizer, expert_balance_sheet, expert_income_statement = load_finetuned_system(data_path=DATA_DIR)
        
        print("\n🎉 All systems are loaded and ready!")
        
        # Create and launch the UI
        ui = create_ui()
        ui.launch()
    else:
        print("\nCould not set up data directory. Exiting application.")
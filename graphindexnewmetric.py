
import os
from datasets import load_dataset
from dotenv import load_dotenv
from langchain_community.vectorstores import Neo4jVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_neo4j import Neo4jGraph
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain.document_loaders import PyPDFLoader
from transformers import pipeline, AutoTokenizer

# Load environment variables
load_dotenv()

# === Load and Chunk PDF ===
pdf_path = "responses.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_documents(pages)

# === Neo4j Configuration ===
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "viratkohli18")

# === Initialize Embeddings and Store in Neo4j ===
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

neo4j_vector = Neo4jVector.from_documents(
    documents,
    hf_embeddings,
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

print("✅ PDF embedded and stored in Neo4j.")

# === LLM for RAG ===
llm = OllamaLLM(model="llama3.2")
qa_chain = RetrievalQA.from_llm(
    llm=llm,
    retriever=neo4j_vector.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# === Load QA Dataset ===
dataset = load_dataset("lavita/medical-eval-sphere")
df = dataset["medical_qa_benchmark_v1.0"].to_pandas()
df = df[["medical_question", "response_a"]].dropna()
test_cases = df.head(50).to_dict("records")

# === Vectara Consistency Classifier ===
classifier = pipeline(
    "text-classification",
    model="vectara/hallucination_evaluation_model",
    tokenizer=AutoTokenizer.from_pretrained("google/flan-t5-base"),
    trust_remote_code=True
)

# === Metric Functions ===
def precision_score(ref, pred):
    ref_tokens = set(ref.strip().lower().split())
    pred_tokens = set(pred.strip().lower().split())
    return len(ref_tokens & pred_tokens) / len(pred_tokens) if pred_tokens else 0.0

def recall_score(ref, pred):
    ref_tokens = set(ref.strip().lower().split())
    pred_tokens = set(pred.strip().lower().split())
    return len(ref_tokens & pred_tokens) / len(ref_tokens) if ref_tokens else 0.0

def f1_score(ref, pred):
    p = precision_score(ref, pred)
    r = recall_score(ref, pred)
    return 2 * (p * r) / (p + r) if (p + r) else 0.0

def calculate_vectara_consistency(question, answer, reference):
    prompt = f"Question: {question}\nAnswer: {answer}\nReference: {reference}"
    result = classifier(prompt)[0]
    return result["label"], result["score"]

# === Evaluation Loop ===
total_f1 = total_precision = total_recall = total_consistency = 0.0
output_lines = []

print("\n📊 Evaluating responses...")

for i, test in enumerate(test_cases):
    question = test["medical_question"]
    expected_answer = test["response_a"]

    response = qa_chain.invoke(question)
    generated_answer = response["result"]
    reference = " ".join([doc.page_content for doc in response["source_documents"]])

    # Compute metrics
    f1 = f1_score(expected_answer, generated_answer)
    precision = precision_score(expected_answer, generated_answer)
    recall = recall_score(expected_answer, generated_answer)
    _, consistency_score = calculate_vectara_consistency(question, generated_answer, reference)

    total_f1 += f1
    total_precision += precision
    total_recall += recall
    total_consistency += consistency_score

    output_lines.extend([
        f"\nQ{i+1}: {question}",
        f"Generated: {generated_answer}",
        f"Expected: {expected_answer}",
        f"F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | Consistency: {consistency_score:.4f}",
        "\nSources:"
    ])
    for j, doc in enumerate(response["source_documents"]):
        output_lines.append(f"Source {j+1}: {doc.page_content[:300]}...")

# === Averages ===
n = len(test_cases)
avg_f1 = total_f1 / n
avg_precision = total_precision / n
avg_recall = total_recall / n
avg_consistency = total_consistency / n

output_lines.append("\n=== AVERAGE METRICS ===")
output_lines.append(f"Average F1 Score: {avg_f1:.4f}")
output_lines.append(f"Average Precision: {avg_precision:.4f}")
output_lines.append(f"Average Recall: {avg_recall:.4f}")
output_lines.append(f"Average Vectara Consistency: {avg_consistency:.4f}")

# === Save Results ===
with open("graph_vector_index_rag_eval_results.txt", "w") as file:
    file.write("\n".join(output_lines))

print("✅ Evaluation complete. Results saved to 'graph_vector_index_rag_eval_results.txt'.")

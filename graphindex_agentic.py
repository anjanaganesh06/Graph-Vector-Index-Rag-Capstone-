
# --------------------------- Imports --------------------------------
import os
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Neo4jVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage
from langchain.schema import HumanMessage, AIMessage, BaseMessage

from typing import TypedDict, Annotated
from typing_extensions import TypedDict
from typing import List, TypedDict, Literal

# ---------------------- Environment & PDF Setup ---------------------
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "viratkohli18")

graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
graph.refresh_schema()

# Load and chunk the PDF
pdf_path = "responses.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_documents(pages)

# Embed and store in Neo4j
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
neo4j_vector = Neo4jVector.from_documents(documents, hf_embeddings, url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

print("✅ PDF chunks stored in Neo4j successfully!")

# ---------------------- LangChain Setup -----------------------------
llm = OllamaLLM(model="deepseek-r1:latest")

qa_chain = RetrievalQA.from_llm(
    llm=llm,
    retriever=neo4j_vector.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)
# ----------------------- Define State -----------------------
class MedicalChatState(TypedDict):
    messages: List[BaseMessage]
    finished: bool

class QAState(TypedDict):
    question: str
    retrieved_docs: list
    answer: str

# ------------------- QA Subgraph Functions -------------------
def get_question(state: QAState) -> QAState:
    return {"question": state["question"]}

def retrieve_docs(state: QAState) -> QAState:
    retriever = neo4j_vector.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(state["question"])
    return {"question": state["question"], "retrieved_docs": docs}

def generate_answer(state: QAState) -> QAState:
    context = " ".join(doc.page_content.replace("\n", " ") for doc in state["retrieved_docs"])
    prompt = f"Question: {state['question']} Context: {context} Answer:"
    result = qa_chain.invoke(prompt)

    # Safely extract the answer text
    if isinstance(result, dict):
        answer = result.get("result") or result.get("output") or str(result)
    else:
        answer = str(result)

    clean_answer = answer.replace("\n", " ").strip()

    return {
        "question": state["question"],
        "retrieved_docs": state["retrieved_docs"],
        "answer": clean_answer
    }


# ------------------- Human Node -------------------
def human_node(state: MedicalChatState) -> MedicalChatState:
    if not state["messages"]:
        print(" MedBot: Welcome! Ask a medical question.")
    else:
        print(" MedBot:", state["messages"][-1].content)

    user_input = input("👤 You: ")
    if user_input.lower() in {"q", "quit", "exit"}:
        return {**state, "finished": True}
    return {
        **state,
        "messages": state["messages"] + [HumanMessage(content=user_input)],
        "finished": False
    }

# ------------------- RAG Wrapper Node -------------------
def rag_wrapper(state: MedicalChatState) -> MedicalChatState:
    question = state["messages"][-1].content
    qa_input = {"question": question}
    rag_state = qa_graph.invoke(qa_input)
    answer = rag_state["answer"]
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=answer)]
    }

# ------------------- Routing Logic -------------------
def decide_next(state: MedicalChatState) -> Literal["chatbot", END]:
    return END if state["finished"] else "chatbot"

def route_back_to_human(_: MedicalChatState) -> Literal["human"]:
    return "human"

# ------------------- Build QA Subgraph -------------------
qa_flow = StateGraph(QAState)
qa_flow.add_node("get_question", RunnableLambda(get_question))
qa_flow.add_node("retrieve", RunnableLambda(retrieve_docs))
qa_flow.add_node("generate", RunnableLambda(generate_answer))

qa_flow.set_entry_point("get_question")
qa_flow.add_edge("get_question", "retrieve")
qa_flow.add_edge("retrieve", "generate")
qa_flow.set_finish_point("generate")

qa_graph = qa_flow.compile()

# ------------------- Main LangGraph -------------------
main_graph_builder = StateGraph(MedicalChatState)
main_graph_builder.add_node("human", human_node)
main_graph_builder.add_node("chatbot", rag_wrapper)

main_graph_builder.set_entry_point("human")
main_graph_builder.add_conditional_edges("human", decide_next)
main_graph_builder.add_conditional_edges("chatbot", route_back_to_human)

main_graph = main_graph_builder.compile()

# ------------------- Run -------------------
# ------------------- Batch Inference Runner -------------------

def run_batch_qa(test_cases: List[dict]):
    results = []

    for idx, case in enumerate(test_cases, 1):
        question = case["medical_question"]
        print(f"\n🔎 Question {idx}: {question}")

        qa_input = {"question": question}
        result = qa_graph.invoke(qa_input)

        answer = result["answer"]
        if isinstance(answer, dict) and "result" in answer:
            clean_answer = answer["result"].strip()
        else:
            clean_answer = str(answer).strip()

        print(f"\n💬 Answer:\n{clean_answer}")

        # ✅ Correct key: retrieved_docs
        docs = result.get("retrieved_docs", [])
        if docs:
            print("\n📄 Sources:")
            for doc in docs:
                metadata = doc.metadata
                page = metadata.get("page", "Unknown")
                source = metadata.get("source", "Unknown")
                print(f"- Page {page} of {source}")

        results.append({
            "question": question,
            "predicted_answer": clean_answer,
            "reference_answer": case.get("response_a", "")
        })

    return results
if __name__ == "__main__":
    from datasets import load_dataset
    import pandas as pd
    import json

    print("📘 Loading dataset...")
    dataset = load_dataset("lavita/medical-eval-sphere")
    df = dataset["medical_qa_benchmark_v1.0"].to_pandas()
    df = df[["medical_question", "response_a"]].dropna()

    # ✅ Use First 50 Questions (you can change .head(50))
    test_cases = df.head(5).to_dict("records")

    print("🤖 Running LangGraph QA on medical questions...")
    results = run_batch_qa(test_cases)

    # ------------------ Metric Utilities ------------------
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    from transformers import pipeline, AutoTokenizer

    classifier = pipeline(
        "text-classification",
        model='vectara/hallucination_evaluation_model',
        tokenizer=AutoTokenizer.from_pretrained('google/flan-t5-base'),
        trust_remote_code=True
    )

    def calculate_bleu(reference, prediction):
        return sentence_bleu(
            [reference.split()],
            prediction.split(),
            smoothing_function=SmoothingFunction().method1
        )

    def calculate_rouge(reference, prediction):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        return scores['rouge1'].fmeasure, scores['rougeL'].fmeasure

    def calculate_vectara_consistency(question, answer, reference):
        prompt = f"Question: {question}\nAnswer: {answer}\nReference: {reference}"
        result = classifier(prompt)[0]
        return result['label'], result['score']

    print("\n📊 Evaluating responses...")

    evaluated_results = []
    for idx, res in enumerate(results, 1):
        question = res["question"]
        prediction = res["predicted_answer"]
        reference = res["reference_answer"]

        bleu = calculate_bleu(reference, prediction)
        rouge1, rougeL = calculate_rouge(reference, prediction)
        vectara_label, vectara_score = calculate_vectara_consistency(question, prediction, reference)

        print(f"\nQ{idx}: {question}")
        print('\n')
        print('\n')
        print(f"🔹 BLEU: {bleu:.4f}, ROUGE-1: {rouge1:.4f}, ROUGE-L: {rougeL:.4f}")
        print(f"🔸 Vectara: {vectara_label} ({vectara_score:.4f})")

        res.update({
            "bleu": bleu,
            "rouge1": rouge1,
            "rougeL": rougeL,
            "vectara_label": vectara_label,
            "vectara_score": vectara_score
        })
        evaluated_results.append(res)

    # ✅ Save Evaluated Results to File
    output_file = "medical_eval_results.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for result in evaluated_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"\n✅ All evaluations saved to: {output_file}")

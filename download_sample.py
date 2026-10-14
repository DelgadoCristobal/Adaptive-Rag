import os
import sys
import json
import ast

from dotenv import load_dotenv
load_dotenv()

# Ensure Windows PowerShell handles UTF-8 correctly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from datasets import load_dataset

def main():
    print("=== Loading EnterpriseRAG-Bench Dataset (onyx-dot-app/EnterpriseRAG-Bench) ===")
    
    # 1. Load questions (small dataset, 500 items)
    print("\n1. Loading questions...")
    try:
        questions_ds = load_dataset("onyx-dot-app/EnterpriseRAG-Bench", "questions")
        test_questions = list(questions_ds["test"])
        print(f"   Successfully loaded {len(test_questions)} test questions.")
    except Exception as e:
        print(f"   [Error] Failed to load questions: {e}")
        return

    # 2. Select first 5 questions and extract their expected_doc_ids
    sample_questions = test_questions[:5]
    print("\n2. Selected Sample Questions for Benchmark:")
    target_doc_ids = set()
    
    for idx, q in enumerate(sample_questions, 1):
        q_id = q.get("question_id")
        q_text = q.get("question")
        q_type = q.get("question_type")
        expected_raw = q.get("expected_doc_ids", "[]")
        
        try:
            expected_ids = ast.literal_eval(expected_raw) if isinstance(expected_raw, str) else expected_raw
        except Exception:
            expected_ids = [expected_raw]
            
        target_doc_ids.update(expected_ids)
        
        print(f"\n   [{idx}] ID: {q_id} ({q_type})")
        print(f"       Question: {q_text}")
        print(f"       Gold Answer: {q.get('gold_answer')}")
        print(f"       Expected Doc IDs: {expected_ids}")

    print(f"\nTarget document IDs to retrieve ({len(target_doc_ids)} docs): {target_doc_ids}")

    # 3. Stream documents to extract the specific expected documents + first few general documents
    print("\n3. Streaming documents to find target documents...")
    try:
        docs_ds = load_dataset("onyx-dot-app/EnterpriseRAG-Bench", "documents", streaming=True)
        doc_iterator = iter(docs_ds["test"])
    except Exception as e:
        print(f"   [Error] Failed to initialize document stream: {e}")
        return

    os.makedirs("data/raw", exist_ok=True)
    
    saved_docs = set()
    total_scanned = 0
    max_scan = 2000  # Scan up to 2000 documents
    
    print(f"   Scanning stream for target documents (up to {max_scan} documents)...")
    for _ in range(max_scan):
        try:
            doc = next(doc_iterator)
            total_scanned += 1
            doc_id = doc.get("doc_id")
            
            # If document matches one of our questions or if we still have few docs and scanned <= 10 docs
            if doc_id in target_doc_ids or (len(saved_docs) < 10 and total_scanned <= 10):
                title = doc.get("title") or doc_id
                source_type = doc.get("source_type", "doc")
                content = doc.get("content", "")
                
                # Format into clean Markdown with title and metadata
                md_text = f"# {title}\n\n**Source:** {source_type}\n**Document ID:** {doc_id}\n\n{content}"
                
                safe_title = "".join([c if c.isalnum() or c in "._-" else "_" for c in title])[:50]
                filename = f"{source_type}_{safe_title}_{doc_id[:8]}.md"
                file_path = os.path.join("data/raw", filename)
                
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(md_text)
                    
                saved_docs.add(doc_id)
                print(f"   ✓ Saved: {filename} (Doc ID: {doc_id}, Type: {source_type})")
                
                # Check if all targets are found
                if target_doc_ids.issubset(saved_docs):
                    print("   ✓ All target question documents found!")
                    break
        except StopIteration:
            break
        except Exception as e:
            print(f"   - Error during streaming: {e}")

    # Save the sample questions into a JSON file for quick reference
    questions_file = "data/sample_questions.json"
    with open(questions_file, "w", encoding="utf-8") as f:
        json.dump(sample_questions, f, indent=2)
    print(f"\n✓ Saved questions to '{questions_file}'")

    print(f"\n=== Done! Saved {len(saved_docs)} documents to 'data/raw' ===")
    print("\nNext steps to run the RAG:")
    print("  1. python main.py ingest")
    print("  2. python main.py index")
    print("  3. python main.py query \"<Question from above>\"")

if __name__ == "__main__":
    main()

import os
import streamlit as st
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

st.title("FactoraIndex: Financial Research")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url.strip():
        urls.append(url.strip())

process_url_clicked = st.sidebar.button("Process URLs")
index_dir = "faiss_index"

main_placeholder = st.empty()

# llm = OllamaLLM(model="gemma:2b", temperature=0.7)
# embeddings = OllamaEmbeddings(model="nomic-embed-text")

try:
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",  
        temperature=0.7
    )
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small" 
    )
    st.sidebar.success("OpenAI API Connected")
except Exception as e:
    st.sidebar.error(f"OpenAI API Error: {str(e)}")
    st.stop()


def load_urls(urls):
    """Load documents from URLs with error handling"""
    docs = []
    for i, url in enumerate(urls):
        try:
            main_placeholder.text(f"Loading URL {i+1}/{len(urls)}: {url[:50]}...")
            loader = WebBaseLoader(web_paths=[url])
            loaded_docs = loader.load()
            docs.extend(loaded_docs)
            main_placeholder.text(f"Loaded URL {i+1}/{len(urls)}")
        except Exception as e:
            st.error(f"Failed to load {url}: {str(e)}")
            continue
    return docs


if process_url_clicked:
    if not urls:
        st.warning("⚠️ Please enter at least one URL")
    else:
        try:
            main_placeholder.text("Data Loading...Started...")
            
            
            docs_raw = load_urls(urls)
            
            if not docs_raw:
                st.error("No documents were loaded. Please check your URLs.")
            else:
                st.success(f"Loaded {len(docs_raw)} documents")
                
               
                main_placeholder.text("Text Splitter...Started...")
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=['\n\n', '\n', '.', ','],
                    chunk_size=1000,
                    chunk_overlap=100  
                )
                
                docs = text_splitter.split_documents(docs_raw)
                st.success(f"Split into {len(docs)} chunks")
                
                
                main_placeholder.text("Embedding Vector Started Building...")
                vectorstore = FAISS.from_documents(docs, embeddings)
                st.success("Vector store created")
                
                
                vectorstore.save_local(index_dir)
                main_placeholder.text("Processing Complete! You can now ask questions.")
                
        except Exception as e:
            st.error(f"❌ Error during processing: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


query = st.text_input("Question: ")

if query:
    if not os.path.exists(index_dir):
        st.warning("⚠️ Please process URLs first before asking questions.")
    else:
        try:
            with st.spinner("Searching for answer..."):
                
                vectorstore = FAISS.load_local(
                    index_dir, embeddings, allow_dangerous_deserialization=True
                )
                retriever = vectorstore.as_retriever()

                
                system_prompt = """You are a financial research assistant.
Use the retrieved context below to answer the question.
If the context does not contain the answer, say you don't know.
Keep answers concise and factual.

Context:
{context}
"""

                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ]
                )

                # Debug: Preview retrieved docs (in expander to avoid clutter)
                # with st.expander("🔍 View Retrieved Context"):
                #     retrieved_docs = retriever.invoke(query)
                #     for i, d in enumerate(retrieved_docs):
                #         st.markdown(f"**Chunk {i+1}:**")
                #         st.write(d.page_content[:500])
                #         st.divider()

                
                question_answer_chain = create_stuff_documents_chain(llm, prompt)
                chain = create_retrieval_chain(retriever, question_answer_chain)

                
                result = chain.invoke({"input": query})

                
                st.header("Answer")
                st.write(result["answer"])

                
                sources = result.get("context", [])
                if sources:
                    st.subheader("Sources:")
                    for doc in sources:
                        if "source" in doc.metadata:
                            st.write(f"- {doc.metadata['source']}")
                            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
import os
from operator import itemgetter
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("Initinalizing Components")

embeddings = OpenAIEmbeddings
llm = ChatOpenAI(model="gpt-6.1-sol")

vectorstore = PineconeVectorStore(
    index_name = os.environ["INDEX_NAME"], embedding=embeddings
)

retriver = vectorstore.as_retriever(search_kwargs={"k":3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
     {context}
      Question: {question}
       Provide detailed answer: """
)

def format_docs(docs):
    """Format retrived documents into single string."""
    return "\n\n".join([doc.page_content for doc in docs])

#Implemention without LCEL

def retrival_chain_without_lcel(query:str):
    """Simple retrival chain without LCEL.
    """

    #Retrive the relevant docs
    docs = retriver.invoke(query)

    #Format the docs into a single string
    context = format_docs(docs)

    #Format the prompt with context and question
    messages = prompt_template.format_messages(context=context, question=query)

    #Invoke the LLM with the formatted message
    response = llm.invoke(messages)

    #return the response content
    return response.content

# IMPLEMENTATION 2: With LCEL (LangChain Expression Language) - BETTER APPROACH

def create_retrival_chain_with_Lcel():
    """Create a retrival chain with LCEL.
    Return a chain that can be invoked with {"question":"..."}
    """
    retrival_chain = (
        RunnablePassthrough.assign(
            context = itemgetter("question") | retriver | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return retrival_chain

if __name__ == "__main__":

    query = "What is Pinecone in machine learning?"

    # ========================================================================
    # Option 0: Raw invocation without RAG
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("=" * 70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(result_raw.content)

    # ========================================================================
    # Option 1: Use implementation WITHOUT LCEL
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Without LCEL")
    print("=" * 70)
    result_without_lcel = retrival_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)

    #Option 2: Use implementation WITH LCEL
    print("\n" + "=" * 70)
    print("Implementaion with LCEL")
    print("- More concise and declarative")
    print("- Built-in streaming: chain.stream()")
    print("- Built-in async: chain.ainvoke()")
    print("- Easy to compose with other chains")
    print("- Better for production use")
    print("=" * 70)

    chain_with_lcel = create_retrival_chain_with_Lcel()
    result_with_Lcel = chain_with_lcel.invoke({"question":query})
    print("\nAnswer:")
    print(result_with_Lcel)

    

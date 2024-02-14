from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader, DirectoryLoader

topics_list = "; ".join(
    [
        # "Relationships and dating",
        "Everyday life and observations",
        "Current events and politics",
        "Pop culture and celebrity gossip",
        "Self-deprecating humor",
        "Satire and social commentary",
    ]
)

text_loader_kwargs = {"autodetect_encoding": True}
loader = DirectoryLoader(
    "./transcripts/",
    show_progress=True,
    loader_cls=TextLoader,
    loader_kwargs=text_loader_kwargs,
)

docs = loader.load()
print(len(docs))

output_parser = StrOutputParser()

llm = ChatOpenAI()

prompt = ChatPromptTemplate.from_template(
    """You are a world class topic retriever. Below is a transcript of a stand-up comedy bit.
Please, analyse provided transcript and identify the main topic. Give me the top three topics from the topic list. Use only the topics from the topic list provided. Write the topics and nothing else.

<topic_list>
{topic_list}
</topic_list>

<video_transcript>
{input}
</video_transcript>

"""
)

chain = prompt | llm | output_parser

print(chain.invoke({"input": docs, "topic_list": topics_list}))

# document_chain = create_stuff_documents_chain(llm, prompt)

# from langchain.chains import create_retrieval_chain

# retriever = vector.as_retriever()
# retrieval_chain = create_retrieval_chain(retriever, document_chain)

# response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
# print(response["answer"])

# LangSmith offers several features that can help with testing:...

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a world class technical documentation writer."),
#         ("user", "{input}"),
#     ]
# )


# print(llm.invoke("how can langsmith help with testing?"))

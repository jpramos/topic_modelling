
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field

class VideoTranscriptTopic(BaseModel):
    topic:str = Field(description="Relevant topic from the video transcript")
    sentiment:str = Field(description="Sentiment of the stand-up bit")

output_parser = PydanticOutputParser(pydantic_object=VideoTranscriptTopic)
format_instructions = output_parser.get_format_instructions()

hf = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-base",
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 4096},
)

text_loader_kwargs = {"autodetect_encoding": True}

loader = DirectoryLoader("./transcripts/", show_progress=True,
        loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)

docs = loader.load()
print(len(docs))

topics_list = "; ".join([
    "Relationships and dating",
    "Everyday life and observations",
    "Current events and politics",
    "Pop culture and celebrity gossip",
    "Self-deprecating humor",
    "Satire and social commentary"])

topic_assignment_msg = '''
Below is a transcript of a stand-up comedy bit.
Please, analyse provided transcript and identify the main topic.

video transcript:
```
{input_data}
```
'''

# messages_template = PromptTemplate.from_template([
#         SystemMessage(content="You're a helpful assistant. Your task is to analyze video transcripts of stand-up comedy"),
#         HumanMessage(content=topic_assignment_msg)
#         ])

messages_template = PromptTemplate.from_template(topic_assignment_msg)

#messages = messages_template.format_template(
#        # topics_list = topics_list,
#        # format_instructions = format_instructions,
#        input_data = docs
#        )

#chat_model = ChatHuggingFace(llm=hf)

chain = messages_template | hf

# response = hf.invoke(messages)

print(chain.invoke({"input_data": docs}))

# print(response)

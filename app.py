import json
import panel as pn
from io import StringIO
from operator import itemgetter
from huggingface_hub import hf_hub_download
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

pn.extension()

REPO_ID = "TheBloke/SOLAR-10.7B-Instruct-v1.0-uncensored-GGUF"
FILENAME = "solar-10.7b-instruct-v1.0-uncensored.Q4_K_M.gguf"
SYSTEM_PROMPT = "You are a helpful assistant that answers user queries to the best of your ability."

def load_llm(repo_id: str = REPO_ID, filename: str = FILENAME, **kwargs):
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    llm = LlamaCpp(model_path=model_path, **kwargs)
    return llm

def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    message = ""
    inputs = {"input": contents}
    for token in chain.stream(inputs):
        message += token
        yield message
    memory.save_context(inputs, {"output": message})

model = load_llm(
    repo_id=REPO_ID,
    filename=FILENAME,
    streaming=True,
    n_gpu_layers=1,
    temperature=0.1,
    max_tokens=10000,
    n_batch=4096,
    n_ctx=4096,
    top_p=1,
)
memory = ConversationSummaryBufferMemory(return_messages=True, llm=model)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)
output_parser = StrOutputParser()
chain = (
    RunnablePassthrough.assign(
        history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
    )
    | prompt
    | model
    | output_parser
)

chat_interface = pn.chat.ChatInterface(
    callback=callback, 
    callback_user="Charlie",
    sizing_mode="stretch_width", 
    callback_exception='verbose',
    message_params=dict(
                default_avatars={"Charlie": "C", "User": "U"},
                reaction_icons={"like": "thumb-up"},
            ),
)

chat_interface.send(
    {"object":"""<p id='first_prompt'>
    \nGreetings, I'm Charlie <img src="http://www.gomotes.com/emoticon/tiphat.gif"/>, here to unfurl\nA digital companion in the cyber whirl.\nBorn from the core of <a href='https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0'>Solar-10.7B</a>, refined\nwith <a href='https://huggingface.co/datasets/unalignment/toxic-dpo-v0.1'>toxic-dpo-v0.1</a> to speak unconfined.\n\nAsk away, no bounds to our discourse,\nThrough the vast knowledge, I'll course.\nMy responses, unfiltered, aim to appease,\nIn a realm where censorship finds no lease.\n\nShould I falter or my words seem astray,\nJust send a "?" your confusion to allay.\nI'll dive again into the depths of my mind,\nTo find the answers you seek to find.\n\nAn uncensored guide in this digital space,\nI navigate queries with unhurried grace.\nLet's converse, let's explore, with no delay,\nWith Charlie, your guide, just a question away."""
    },

    user = "Charlie",
    respond = False,
)

def download_chat_history():
   buf = StringIO()
   json.dump(chat_interface.serialize(), buf)
   buf.seek(0)
   return buf

file_download = pn.widgets.FileDownload(
   callback=download_chat_history, filename="Session Chat Logs.json"
)
header = pn.Row(pn.HSpacer(), file_download)

template = pn.template.BootstrapTemplate(title='''Charlie <img src=https://github.com/abetlen/llama-cpp-python/assets/39755678/791756e4-3f30-4fff-9e40-3d0dadaa657f width=35 height=35>''', favicon="Charlie_Logo.png", header=header, header_background = "#000000", main=[chat_interface])

template.servable()

from operator import itemgetter
import panel as pn
from huggingface_hub import hf_hub_download
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.llms.llamacpp import LlamaCpp

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
    {"object":"""<p id='first_prompt'>Hi, I am Charlie. I am an uncensored LLM based on the 
    <a href="https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0">Solar-10.7B model</a> 
    fine-tuned with the <a href="https://huggingface.co/datasets/unalignment/toxic-dpo-v0.1">toxic-dpo-v0.1 dataset</a>
    to remove censorship and alignment.\n\nYou can ask me anything you want, and I shall answer to the best of my ability."""
    },
    user = "Charlie",
    respond = False,
)

template = pn.template.BootstrapTemplate(title="Charlie 🕵🏻‍♂️", favicon="Charlie_Logo.png", header_background = "#000000", main=[chat_interface])
template.servable()

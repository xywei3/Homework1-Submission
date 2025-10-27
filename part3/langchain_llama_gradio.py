from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama 
import gradio as gr

def captial_city(country):
    prompt = PromptTemplate.from_template(
        "What is the capital of {topic}?"
    )
    # Define the model
    model = ChatOllama(model = "llama2")  # Using Ollama 
    # Chain the components together using LCEL
    chain = (
        # LCEL syntax: use the pipe operator | to connect each step
        {"topic": RunnablePassthrough()}  # Accept user input
        | prompt                          # Transform it into a prompt message
        | model                           # Call the model
        | StrOutputParser()               # Parse the output as a string
    )
    return chain.invoke(country)

demo = gr.Interface(
    fn=captial_city, 
    inputs="textbox", 
    outputs="textbox"
)

demo.launch()
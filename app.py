import os
import uuid

import replicate
import requests
import streamlit as st
from streamlit.logger import get_logger

user_session_id = uuid.uuid4()

logger = get_logger(__name__)
st.session_state.disabled = False
st.title("Llama-v2 Chat Demo with Message History")
st.markdown(
    "Built by [Nirant Kasliwal](https://nirantk.com/about/). Sponsored by [The GenerativeAI Community 🇮🇳](https://nirantk.com/community)"
)
os.environ["REPLICATE_API_TOKEN"] = st.secrets["REPLICATE_API_TOKEN"]
fastapi_endpoint = st.secrets["FASTAPI_ENDPOINT"]
secret_token = st.secrets[
    "SECRET_TOKEN"
]  # Token for FastAPI endpoint authentication to log queries

llama_family = {
    "Llama7B-v2-Chat": "a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea",
    "Llama13B-v2-Chat": "a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
    "Llama70B-v2-Chat": "replicate/llama70b-v2-chat:e951f18578850b652510200860fc4ea62b3b16fac280f83ff32282f87bbd2e48",
}
model_choice = st.selectbox("Model", options=tuple(llama_family.keys()))
llm_model = llama_family[model_choice]
st.session_state["llm_model"] = llm_model
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("## Model Parameters")
    # repetition_penalty = st.slider("Repetition Penalty", 0.0, 2.0, 1.0, help="Penalty for repeated words in generated text; 1 is no penalty, values greater than 1 discourage repetition, less than 1 encourage it.")
    temperature = st.slider(
        "Temperature",
        0.01,
        5.0,
        0.9,
        help="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.",
    )
    max_tokens = st.slider(
        "Max Length in Number of Tokens",
        10,
        500,
        100,
        help="Maximum number of tokens to generate. A word is generally 2-3 tokens",
    )
    top_p = st.slider(
        "Top P",
        0.01,
        1.0,
        0.2,
        help="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def on_select():
    st.session_state.disabled = True


def get_message_history():
    for message in st.session_state.messages:
        role, content = message["role"], message["content"]
        yield f"{role.title()}: {content}"


if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Include System Prompt
    system_prompt = """You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
    # st.session_state.messages.append({"role": "system", "content": system_prompt})
    # with st.chat_message("system"):
    #     st.markdown(system_prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        message_history = "\n".join(list(get_message_history())[-3:])
        # logger.info(f"Get message history func: {list(get_message_history())}")

        # Combine user prompt and system prompt with the generation stopper
        combined_prompt = f"[INST]<<SYS>>{system_prompt}<<SYS>>\n\n{message_history}\n\nAssitant:[/INST]"
        output = replicate.run(
            llm_model,
            input={
                "prompt": combined_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                # "repetition_penalty": repetition_penalty,
                "debug": True,
            },
        )

        # Extract the assistant's response until the generation stopper
        for item in output:
            if "<<SYS>>" in item:
                full_response = item.split("<<SYS>>")[0]
                logger.info(f"System prompt detected")
                break
            else:
                full_response += item
            message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

    # Sentiment Radio Button
    response_sentiment = st.radio(
        "How was the Assistant's response?",
        ["😁", "😕", "😢"],
        key="response_sentiment",
        disabled=st.session_state.disabled,
        horizontal=True,
        index=1,
        help="This helps us improve the model.",
        # hide the radio button on click
        on_change=on_select(),
    )
    logger.info(f"{user_session_id} | {full_response} | {response_sentiment}")

    # Logging to FastAPI Endpoint
    headers = {"Authorization": f"Bearer {secret_token}"}
    log_data = {"log": f"{user_session_id} | {full_response} | {response_sentiment}"}
    response = requests.post(
        fastapi_endpoint, json=log_data, headers=headers, timeout=10
    )
    if response.status_code == 200:
        logger.info("Query logged successfully")

    st.session_state.messages.append({"role": "assistant", "content": full_response})

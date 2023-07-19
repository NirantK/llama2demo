import os
import replicate
import streamlit as st
import uuid

user_session_id = uuid.uuid4()

logger = get_logger(__name__)
st.session_state.disabled = False
st.title("Llama-v2 Chat Demo with Message History")
st.markdown(
    "Built by [Nirant Kasliwal](https://nirantk.com/about/). Sponsored by [The GenerativeAI Community 🇮🇳](https://nirantk.com/community)"
)
os.environ["REPLICATE_API_TOKEN"] = st.secrets["REPLICATE_API_TOKEN"]
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

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        message_history = "\n".join(list(get_message_history())[-3:])
        logger.info(f"{user_session_id} Message History: {message_history}")
        output = replicate.run(
            llm_model,
            input={
                "prompt": f"{message_history}\nAssistant:",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                # "repetition_penalty": repetition_penalty,
                "debug": True,
            },
        )
    for item in output:
        full_response += item
        message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    response_sentiment = st.radio(
        "How was the Assistant's response?",
        ["😁", "😕", "😢"],
        key="response_sentiment",
        disabled=st.session_state.disabled,
        horizontal=True,
        help="This helps us improve the model.",
        # hide the radio button on click
        on_change=on_select(),
    )
    logger.info(f"{user_session_id} Response Sentiment: {response_sentiment}")
    st.session_state.messages.append({"role": "assistant", "content": full_response})

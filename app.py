import os
import replicate
import streamlit as st

st.title("Llama13b-v2 Chat Demo")

os.environ["REPLICATE_API_TOKEN"] = st.secrets["REPLICATE_API_TOKEN"]
st.sidebar.markdown(
    "Built with [Replicate](https://replicate.com), a platform for running AI models on live data by [Nirant Kasliwal](https://nirantk.com/about/)"
)
st.session_state[
    "llm_model"
] = "a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        output = replicate.run(
            "a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
            input={
                "prompt": prompt,
            },
        )
    # The a16z-infra/llama13b-v2-chat model can stream output as it's running.
    # The predict method returns an iterator, and you can iterate over that output.
    for item in output:
        # https://replicate.com/a16z-infra/llama13b-v2-chat/versions/df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5/api#output-schema
        print(item)
        # for response in openai.ChatCompletion.create(
        #     model=st.session_state["llm_model"],
        #     messages=[
        #         {"role": m["role"], "content": m["content"]}
        #         for m in st.session_state.messages
        #     ],
        #     stream=True,
        # ):
        full_response += item
        message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

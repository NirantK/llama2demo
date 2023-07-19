import os
import replicate
import streamlit as st

st.title("Llama-v2 Chat Demo with Message History")
st.markdown("Built by [Nirant Kasliwal](https://nirantk.com/about/)")
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
        message_history = "\n".join(
            [
                f"{message:['role']}: {message['content']}"
                for message in st.session_state.messages
            ]
        )
        print("Message History:", message_history)
        output = replicate.run(
            llm_model,
            input={
                "prompt": f"{message_history}\nUser: {prompt}\nAssistant:",
                "max_tokens": 100,
                "temperature": 0.1,
                "top_p": 0.1,
                "debug": True,
            },
        )
    for item in output:
        full_response += item
        message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

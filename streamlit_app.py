import streamlit as st
from transcript import final_transcript
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core import ServiceContext
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    load_index_from_storage,
)
import os
from openai import OpenAI

import os
from dotenv import load_dotenv

load_dotenv()
openai_key = os.getenv('OPENAI_API_KEY')

os.environ["OPENAI_API_KEY"] = (openai_key)

import toml
primaryColor = toml.load(".streamlit/config.toml")['theme']['primaryColor']
custom_css = f"""
<style>
body {{
    background-color: #000000;
    color: white;
}}
div.stButton > button {{
    background: linear-gradient(90deg, #01e8df, #1eff80);
    color:black;
}}
div.stButton > button:hover {{
    background: linear-gradient(80deg, #01e8df, #1eff80);
    color:black;
    border : 1px solid #000000;
}}

div.stButton {{
    text-align: center;
}}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)
st.image("logo.png", width=30)  # Make sure to have the 'logo.png' in the correct path

def main():
    st.title("AI-powered search engine demo for Sparx Podcast.")

    user_input = st.text_input("Ask a question:" , "What is zugzwang")

    if st.button("Submit"):
        st.write(f"Query: {user_input}")

        lc_embed_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        embed_model = LangchainEmbedding(lc_embed_model)
        service_context = ServiceContext.from_defaults(embed_model=embed_model)

        PERSISTS_DIR = "./vishanathan-anand-and-mukesh-bansal"
        if not os.path.exists(PERSISTS_DIR):
            documents = [
                Document(
                    text=f"{d['speaker']} : {d['text']}",
                    metadata={"start": d["start"], "end": d["end"]},
                )
                for d in final_transcript
            ]
            multi_doc_index = VectorStoreIndex.from_documents(
                documents, service_context=service_context
            )
            multi_doc_index.storage_context.persist(persist_dir=PERSISTS_DIR)
        else:
            storage_context = StorageContext.from_defaults(
                persist_dir=PERSISTS_DIR)
            multi_doc_index = load_index_from_storage(
                storage_context, service_context=service_context
            )

        multi_doc_query = multi_doc_index.as_retriever(similarity_top_k=3)

        client = OpenAI()

        def generate_openai_response(prompt):
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"Todo: \n Read the whole transcript provided. Answer in 3-4 line about the question asked. Answer only from the transcript provided and who mentioned the topic \n{prompt}",
                    }
                ],
                model="gpt-4o",
            )
            return response.choices[0].message.content

        # Generate a response based on the document content
        document_text = " ".join([d["text"] for d in final_transcript])
        openai_response = generate_openai_response(
            f"Question: \n  {user_input} \n Podcast:\n For the latest episode of SparX, Mukesh Bansall, Founder of Myntra and Cult.fit, is in conversation with India's first Grandmaster, Viswanathan Anand. Viswanathan is a five-time World Chess Champion. Anand's most significant achievements include being a five-time World Chess Champion, winning the World Rapid Chess Championship, and the World Blitz Chess Championship. Additionally, his victories in prestigious tournaments like Linares and Corus are highly notable in the chess world. Viswanathan revolutionised the chess landscape in India. His victories brought national pride and global recognition, inspiring a surge in young players and increased investment in chess infrastructure. Anand's success fostered a vibrant chess culture, making India a formidable force in the sport. Join us for an inspiring conversation with the Grandmaster on his journey with chess, some stories with other chess players, the evolution of chess, and his experiences throughout his career. \n transcript:\n{document_text}"
        )
        st.write("Summary: ", openai_response)

        # Step 2: Use the response as a query for the multi-document index
        multi_doc_answer = multi_doc_query.retrieve(openai_response)

        # Display the results from the multi-document index in the Streamlit UI
        st.write("Youtube Timestamps:")
        for r in multi_doc_answer:
            # st.write(f"Score: {r.score}")
            start_time_seconds = (
                r.metadata["start"] // 1000
            )  # Convert milliseconds to seconds
            youtube_link = (
                f"https://www.youtube.com/watch?v=6-RtRVIjlkQ&t={start_time_seconds}"
            )
            if r.score > 0.5:
                st.write(f"Link: {youtube_link}")


if __name__ == "__main__":
    main()

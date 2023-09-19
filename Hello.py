import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfFileReader, PdfFileWriter, PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

import pickle
import os
# Load api key lib
from dotenv import load_dotenv
import base64

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image
import io
import pdf2image



load_dotenv()

# Background images add function
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"jpeg"};base64,{encoded_string.decode()});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def generate_pdf(user_text):
    # Create a PDF buffer using StringIO
    pdf_buffer = io.BytesIO()

    # Create a new PDF document
    c = canvas.Canvas(pdf_buffer, pagesize=letter)

    # Add user text to the PDF
    c.drawString(50, 600, user_text)

    # Save the PDF to the buffer
    c.save()

    # Reset the buffer position
    pdf_buffer.seek(0)
    return pdf_buffer

def main():
    st.header("📄Summarize Insurance Docs 🤗")

    # upload a your pdf file
    pdf = st.file_uploader("Step 1 - Upload your PDF", type='pdf')
   

    if pdf is not None:
        # st.write(pdf.name)
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # langchain_textspliter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text=text)

        # store pdf name
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                vectorstore = pickle.load(f)
            # st.write("Already, Embeddings loaded from the your folder (disks)")
        else:
            # embedding (Openai methods)
            embeddings = OpenAIEmbeddings()

            # Store the chunks part in db (vector)
            vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vectorstore, f)

            # st.write("Embedding computation completed")

        # st.write(chunks)

        st.markdown("""---""")
        # Accept user questions/query
        left_column, right_column = st.columns(2)

        with left_column:
            query = st.text_input("Step 2 - Ask questions about the information you need.", "Please summarize this policy")
            query = query + 'Add new line characters in the proper places'
            query = query + "Always provide page numbers to reference where you found the information"
            if query:
                docs = vectorstore.similarity_search(query=query, k=3)
                # openai rank lnv process
                llm = OpenAI(temperature=0)
                chain = load_qa_chain(llm=llm, chain_type="stuff")

                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=query)
                    print(cb)
                st.write(":blue[AI Response]")
                st.markdown(response)                
        with right_column:
            st.title("PDF Generator")

            # User inputs
            user_text = st.text_area("Step 3) Copy Information To Here", "This will be used to create a printable document.")

            if st.button("Generate PDF"):
                if user_text:
                    # Generate the PDF
                    pdf_buffer = generate_pdf(user_text)
                    if pdf_buffer:
                        st.write("### Generated PDF:")
                        
                    # st.markdown(
                    #     f'<a href="data:application/pdf;base64,{pdf_buffer.getvalue().encode("base64").decode()}" download="output.pdf">Download PDF</a>',
                    #     unsafe_allow_html=True
                    # )

                    # Offer the PDF download
                    st.download_button("Download PDF", pdf_buffer, file_name="generated_pdf.pdf", key="pdf-download")
                else:
                    st.error("Please enter some text.")

            

            

     


if __name__ == "__main__":
    main()

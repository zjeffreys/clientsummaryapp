import streamlit as st
import openai

# Set your OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]
# openai.api_key="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"

# Create a Streamlit app
def main():
    st.title("Streamlit OpenAI Demo")
    st.write("Use the OpenAI API to generate text!")

    # Input text box for user prompt
    user_prompt = st.text_area("Enter a prompt:", "Once upon a time,")

    if st.button("Generate Text"):
        if user_prompt:
            try:
                # Use the OpenAI API to generate text
                response = openai.Completion.create(
                    engine="davinci",
                    prompt=user_prompt,
                    max_tokens=50,  # Adjust as needed
                    n = 1,
                    stop=None
                )
                generated_text = response.choices[0].text

                # Display the generated text
                st.write("Generated Text:")
                st.write(generated_text)

            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter a prompt.")

if __name__ == "__main__":
    main()

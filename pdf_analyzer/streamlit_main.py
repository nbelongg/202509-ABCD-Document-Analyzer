import streamlit as st
import yaml
import streamlit_authenticator as stauth
from streamlit_app.streamlit_utils import pdf_analyzer_ui, pdf_prompt_selection


@st.cache_data
def load_config():
    with open('config.yml') as file:
        return yaml.load(file, Loader=yaml.SafeLoader)


def main():
    if 'authentication_status' not in st.session_state:
        st.session_state['authentication_status'] = None

    config = load_config()

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'])

    if st.session_state['authentication_status'] is None:
        authenticator.login()

    if st.session_state['authentication_status']:
        st.sidebar.title(f"Welcome {st.session_state['name']}")
        authenticator.logout('Logout', 'sidebar')
        pdf_analyzer_ui()
        #pdf_prompt_selection()
    elif st.session_state['authentication_status'] is False:
        st.error('Username/password is incorrect')
    elif st.session_state['authentication_status'] is None:
        st.warning('Please enter your username and password')


if __name__ == "__main__":
    main()